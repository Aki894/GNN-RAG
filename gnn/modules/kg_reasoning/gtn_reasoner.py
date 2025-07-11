import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_gnn import BaseGNNLayer
from .gtn_layer import GTNLayer

VERY_NEG_NUMBER = -100000000000

class GTNReasonLayer(BaseGNNLayer):
    """
    混合图谱推理框架 (Hybrid-Graph Reasoning Framework)
    
    结合原始知识图谱和由GTN生成的元路径图，实现更强大的推理能力。
    """
    def __init__(self, args, num_entity, num_relation, entity_dim, alg):
        super(GTNReasonLayer, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.alg = alg
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        
        # GTN相关参数
        self.use_gtn = args.get('use_gtn', True)  # 默认启用GTN
        self.gtn_channels = args.get('gtn_channels', 2)  # 元路径通道数
        self.gtn_layers = args.get('gtn_layers', 1)  # GTN层数
        
        # 原始图的处理参数
        self.use_posemb = args['pos_emb']
        self.use_residual = args.get('use_residual', False)
        self.use_node_adaptive_residual = args.get('use_node_adaptive_residual', False)
        
        self.init_layers(args)
        
        # 创建GTN层
        if self.use_gtn:
            self.gtn_modules = nn.ModuleList([
                GTNLayer(args, num_entity, num_relation, entity_dim, num_channels=self.gtn_channels)
                for _ in range(self.gtn_layers)
            ])
            
            # 混合图谱融合层
            self.hybrid_fusion = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(entity_dim * 2, entity_dim),
                    nn.Sigmoid()
                ) for _ in range(self.num_gnn)
            ])
            
            # 最终输出变换
            self.output_transform = nn.ModuleList([
                nn.Linear(entity_dim, entity_dim) 
                for _ in range(self.num_gnn)
            ])

    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.glob_lin = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        self.lin = nn.Linear(in_features=2*entity_dim, out_features=entity_dim)
        assert self.alg == 'bfs'
        self.linear_dropout = args['linear_dropout']
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        
        if self.use_residual or self.use_node_adaptive_residual:
            self.res_norm = nn.LayerNorm(entity_dim)
            if self.use_node_adaptive_residual:
                self.node_adaptive_linear = nn.Linear(in_features=entity_dim, out_features=1)

        for i in range(self.num_gnn):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            if self.alg == 'bfs':
                self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2*(self.num_ins)*entity_dim + entity_dim, out_features=entity_dim))

            if self.use_posemb:
                self.add_module('pos_emb' + str(i), nn.Embedding(self.num_relation, entity_dim))
                self.add_module('pos_emb_inv' + str(i), nn.Embedding(self.num_relation, entity_dim))
        self.lin_m = nn.Linear(in_features=(self.num_ins)*entity_dim, out_features=entity_dim)

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, rel_features_inv, query_entities, query_node_emb=None):
        self.local_entity = local_entity
        self.kb_adj_mat = kb_adj_mat
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity

        # 初始化GTN模块的local_entity
        if self.use_gtn:
            for gtn_module in self.gtn_modules:
                gtn_module.init_local_entity(local_entity)

        # 转换kb_adj_mat的元素为PyTorch张量，然后赋值给self.edge_list
        h, r, t, b_ids, f_ids, weights, _ = kb_adj_mat
        
        h_tensor = torch.LongTensor(h).to(self.device)
        r_tensor = torch.LongTensor(r).to(self.device)
        t_tensor = torch.LongTensor(t).to(self.device)
        b_ids_tensor = torch.LongTensor(b_ids).to(self.device)
        f_ids_tensor = torch.LongTensor(f_ids).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device) # weights 通常是浮点数

        self.edge_list = (h_tensor, r_tensor, t_tensor, b_ids_tensor, f_ids_tensor, weights_tensor, None)

        # 初始化GTN模块的边列表信息
        if self.use_gtn:
            for gtn_module in self.gtn_modules:
                gtn_module.init_edge_list(self.edge_list, batch_size, max_local_entity)

        self.rel_features = rel_features
        self.rel_features_inv = rel_features_inv
        self.local_entity_emb = local_entity_emb
        self.possible_cand = []
        
        # 现在self.edge_list已经包含张量，build_matrix将正常工作
        self.build_matrix()
        
        self.query_entities = query_entities
        self.query_node_emb = query_node_emb
        
        # 这些字段在reason_layer和reason_layer_inv中也会用到，确保它们是张量
        # 它们现在可以直接引用self.edge_list中的张量元素
        self.batch_heads_list = h_tensor
        self.batch_rels_list = r_tensor
        self.batch_tails_list = t_tensor
        self.batch_ids_list = b_ids_tensor
        self.fact_ids_list = f_ids_tensor
        self.weight_list = weights_tensor

    def reason_layer(self, curr_dist, instruction, rel_linear, pos_emb):
        """
        在原始图上聚合邻居表示（非稀疏矩阵版本）
        """
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        rel_features = self.rel_features
        
        # 初始化结果张量
        neighbor_rep = torch.zeros(batch_size, max_local_entity, self.entity_dim).to(self.device)
        
        # 对每个批次分别处理
        for b in range(batch_size):
            # 获取当前批次的实体分布
            curr_dist_b = curr_dist[b]  # [max_local_entity]
            
            # 对每个关系类型分别处理
            for r in range(self.num_relation):
                # 找出当前批次和关系类型的所有边
                mask = (self.batch_rels_list == r) & (self.batch_ids_list == b)
                if not mask.any():
                    continue
                
                # 获取头尾实体
                heads = self.batch_heads_list[mask]
                tails = self.batch_tails_list[mask]
                
                # 获取关系表示
                rel_emb = rel_features[r]  # [entity_dim]
                
                # 应用位置嵌入（如果有）
                if pos_emb is not None:
                    pe = pos_emb.weight[r]  # [entity_dim]
                    rel_transformed = rel_linear(rel_emb) + pe
                else:
                    rel_transformed = rel_linear(rel_emb)
                
                # 获取问题指令
                query_emb = instruction[b]  # [entity_dim]
                
                # 计算关系-查询交互
                fact_val = F.relu(rel_transformed * query_emb)  # [entity_dim]
                
                # 对每条边进行消息传递，添加边界检查
                for h, t in zip(heads, tails):
                    # 确保索引在有效范围内
                    if h >= max_local_entity or t >= max_local_entity:
                        continue
                    
                    # 从头实体到尾实体传递消息
                    # 使用头实体的概率作为权重
                    weight = curr_dist_b[h].item()
                    neighbor_rep[b, t] += fact_val * weight
        
        return neighbor_rep

    def reason_layer_inv(self, curr_dist, instruction, rel_linear, pos_emb_inv):
        """
        在原始图上聚合反向邻居表示（非稀疏矩阵版本）
        """
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        rel_features = self.rel_features_inv
        
        # 初始化结果张量
        neighbor_rep = torch.zeros(batch_size, max_local_entity, self.entity_dim).to(self.device)
        
        # 对每个批次分别处理
        for b in range(batch_size):
            # 获取当前批次的实体分布
            curr_dist_b = curr_dist[b]  # [max_local_entity]
            
            # 对每个关系类型分别处理
            for r in range(self.num_relation):
                # 找出当前批次和关系类型的所有边
                mask = (self.batch_rels_list == r) & (self.batch_ids_list == b)
                if not mask.any():
                    continue
                
                # 获取头尾实体（反向传播，所以头尾交换）
                heads = self.batch_heads_list[mask]
                tails = self.batch_tails_list[mask]
                
                # 获取关系表示
                rel_emb = rel_features[r]  # [entity_dim]
                
                # 应用位置嵌入（如果有）
                if pos_emb_inv is not None:
                    pe = pos_emb_inv.weight[r]  # [entity_dim]
                    rel_transformed = rel_linear(rel_emb) + pe
                else:
                    rel_transformed = rel_linear(rel_emb)
                
                # 获取问题指令
                query_emb = instruction[b]  # [entity_dim]
                
                # 计算关系-查询交互
                fact_val = F.relu(rel_transformed * query_emb)  # [entity_dim]
                
                # 对每条边进行消息传递（反向），添加边界检查
                for h, t in zip(heads, tails):
                    # 确保索引在有效范围内
                    if h >= max_local_entity or t >= max_local_entity:
                        continue
                        
                    # 从尾实体到头实体传递消息
                    # 使用尾实体的概率作为权重
                    weight = curr_dist_b[t].item()
                    neighbor_rep[b, h] += fact_val * weight
        
        return neighbor_rep

    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        """
        在混合图谱上进行推理
        
        Args:
            current_dist: 当前实体概率分布
            relational_ins: 问题指令表示
            step: 当前GNN层索引
            return_score: 是否返回分数
            
        Returns:
            current_dist: 更新后的实体概率分布
            local_entity_emb: 更新后的实体表示
        """
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        score_func = self.score_func
        neighbor_reps = []
        
        if self.use_posemb:
            pos_emb = getattr(self, 'pos_emb' + str(step))
            pos_emb_inv = getattr(self, 'pos_emb_inv' + str(step))
        else:
            pos_emb, pos_emb_inv = None, None

        # 存储当前层的输入，用于残差连接
        h_v_l_minus_1 = self.local_entity_emb

        # 1. 在原始图上进行消息传递
        for j in range(relational_ins.size(1)):
            # 正向和反向关系的消息传递
            neighbor_rep = self.reason_layer(current_dist, relational_ins[:,j,:], rel_linear, pos_emb)
            neighbor_reps.append(neighbor_rep)

            neighbor_rep = self.reason_layer_inv(current_dist, relational_ins[:,j,:], rel_linear, pos_emb_inv)
            neighbor_reps.append(neighbor_rep)

        neighbor_reps = torch.cat(neighbor_reps, dim=2)
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_reps), dim=2)
        
        # 应用原始图的变换
        original_transformed_h_v = F.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))
        
        # 2. 如果启用GTN，则在元路径图上进行消息传递
        if self.use_gtn:
            try:
                # 从问题指令中提取全局表示
                question_emb = torch.mean(relational_ins, dim=1)  # [batch_size, entity_dim]
                
                # 使用GTN生成元路径图并进行消息传递
                meta_path_h_v = self.gtn_modules[0].forward(
                    self.local_entity_emb, 
                    question_emb, 
                    current_dist
                )
                
                # 3. 混合图谱融合
                # 计算融合门控值
                concat_emb = torch.cat([original_transformed_h_v, meta_path_h_v], dim=2)
                gate = self.hybrid_fusion[step](concat_emb)
                
                # 门控融合
                fused_h_v = gate * meta_path_h_v + (1 - gate) * original_transformed_h_v
                
                # 最终变换
                transformed_h_v = self.output_transform[step](fused_h_v)
            except Exception as e:
                # 打印详细的错误信息
                print(f"GTN处理出错，回退到原始图结果: {e}")
                import traceback
                traceback.print_exc() # 打印完整的堆栈跟踪
                transformed_h_v = original_transformed_h_v
        else:
            # 不使用GTN时，直接使用原始图的结果
            transformed_h_v = original_transformed_h_v
        
        # 应用残差连接
        if self.use_residual:
            self.local_entity_emb = self.res_norm(h_v_l_minus_1 + transformed_h_v)
        elif self.use_node_adaptive_residual:
            # 节点自适应残差连接
            gate = torch.sigmoid(self.node_adaptive_linear(h_v_l_minus_1))
            self.local_entity_emb = self.res_norm(h_v_l_minus_1 + gate * transformed_h_v)
        else:
            self.local_entity_emb = transformed_h_v

        # 计算分数和概率分布
        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        
        if return_score:
            return score_tp, current_dist
        
        return current_dist, self.local_entity_emb 