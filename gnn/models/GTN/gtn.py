import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel
from modules.kg_reasoning.graph_transformer import GraphTransformerNetwork
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder, Fusion, QueryReform

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000

class GTN(BaseModel):
    """
    Graph Transformer Network模型
    基于论文《Graph Transformer Networks: Learning Meta-path Graphs to Improve GNNs》
    """
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        初始化GTN模型
        """
        super(GTN, self).__init__(args, num_entity, num_relation, num_word)
        self.norm_rel = args['norm_rel']
        self.layers(args)
        
        self.loss_type = args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.lm = args['lm']
        
        # GTN特有参数
        self.num_heads = args.get('num_heads', 8)
        self.num_gtn_layers = args.get('num_gtn_layers', 2)
        self.d_ff = args.get('d_ff', 4 * self.entity_dim)
        self.gtn_dropout = args.get('gtn_dropout', 0.1)
        
        # 将所需参数添加到args
        args['num_heads'] = self.num_heads
        args['num_gtn_layers'] = self.num_gtn_layers
        args['d_ff'] = self.d_ff
        args['gtn_dropout'] = self.gtn_dropout
        
        self.private_module_def(args, num_entity, num_relation)
        
        self.to(self.device)
        
        # 融合和更新模块
        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))

    def layers(self, args):
        """
        初始化各种层
        """
        # 初始化实体嵌入
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim

        self.linear_dropout = args['linear_dropout']
        
        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        
        # dropout层
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                      linear_drop=self.linear_drop, device=self.device, norm_rel=self.norm_rel)

        self.self_att_r = AttnEncoder(self.entity_dim)
        
        # 损失函数
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        """
        初始化实体嵌入
        """
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                             edge_list=kb_adj_mat,
                                             rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)
            local_entity_emb = self.entity_linear(local_entity_emb)
        
        return local_entity_emb
    
    def get_rel_feature(self):
        """
        获取关系特征
        """
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)
        else:
            rel_features = self.instruction.question_emb(self.rel_features)
            rel_features_inv = self.instruction.question_emb(self.rel_features_inv)
            
            rel_features = self.self_att_r(rel_features, (self.rel_texts != self.instruction.pad_val).float())
            rel_features_inv = self.self_att_r(rel_features_inv, (self.rel_texts != self.instruction.pad_val).float())
            if self.lm == 'lstm':
                rel_features = self.self_att_r(rel_features, (self.rel_texts != self.num_relation+1).float())
                rel_features_inv = self.self_att_r(rel_features_inv, (self.rel_texts_inv != self.num_relation+1).float())

        return rel_features, rel_features_inv

    def private_module_def(self, args, num_entity, num_relation):
        """
        定义模型私有模块：LM编码器，GNN等
        """
        # 初始化实体嵌入
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        
        # 使用Graph Transformer Network作为推理层
        self.reasoning = GraphTransformerNetwork(args, num_entity, num_relation, entity_dim)
        
        # 文本编码器
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        """
        初始化推理过程
        """
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        rel_features, rel_features_inv = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
        self.init_entity_emb = self.local_entity_emb
        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist
        
        self.reasoning.init_reason(
            local_entity=local_entity,
            kb_adj_mat=kb_adj_mat,
            local_entity_emb=self.local_entity_emb,
            rel_features=rel_features,
            rel_features_inv=rel_features_inv,
            query_entities=query_entities
        )

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        """
        计算损失
        """
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def forward(self, batch, training=False):
        """
        前向传播函数：创建指令并执行GNN推理
        """
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input = torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val
            query_mask = (q_input != pad_val).float()
        else:
            query_mask = (q_input != self.num_word).float()

        """
        指令生成
        """
        self.init_reason(
            curr_dist=current_dist, 
            local_entity=local_entity,
            kb_adj_mat=kb_adj_mat, 
            q_input=q_input, 
            query_entities=query_entities
        )
        
        self.instruction.init_reason(q_input)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i) 
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        
        self.dist_history.append(self.curr_dist)

        """
        GTN推理
        """
        for t in range(self.num_iter):
            # 整合所有指令
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            self.curr_dist = current_dist
            
            # 应用Graph Transformer层
            for j in range(self.num_gnn):
                self.curr_dist, global_rep = self.reasoning(self.curr_dist, relation_ins, step=j)
                
            self.dist_history.append(self.curr_dist)
            qs = []

            """
            指令更新
            """
            for j in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j))
                q = reform(self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                qs.append(q.unsqueeze(1))
                self.instruction.instructions[j] = q.unsqueeze(1)
        
        """
        答案预测
        """
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        
        # 计算损失
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        
        # 选择最可能的答案
        pred = torch.max(pred_dist, dim=1)[1]
        
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
            
        return loss, pred, pred_dist, tp_list 