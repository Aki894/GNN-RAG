import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import MultiheadAttention

from .base_gnn import BaseGNNLayer

VERY_NEG_NUMBER = -100000000000

class GraphAttentionLayer(nn.Module):
    """
    图注意力层
    """
    def __init__(self, in_features, out_features, num_heads, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * num_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        """
        x: 节点特征矩阵 [batch_size, num_nodes, in_features]
        adj: 邻接矩阵 [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = x.size()
        
        # 线性变换
        Wh = torch.matmul(x, self.W)  # [batch_size, num_nodes, num_heads * out_features]
        Wh_reshaped = Wh.view(batch_size, num_nodes, self.num_heads, self.out_features)
        
        # 简化版本的注意力计算 - 使用邻接矩阵作为注意力权重
        attention = adj.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
        
        # 归一化注意力权重 - 按行归一化
        attention_sum = torch.sum(attention, dim=2, keepdim=True)
        # 防止除以零
        attention_sum = torch.clamp(attention_sum, min=1.0)
        attention = attention / attention_sum
        
        # 对每个头使用高效的矩阵乘法进行消息传递
        output = torch.zeros_like(Wh_reshaped)
        
        for h in range(self.num_heads):
            # 提取当前头的注意力权重 [batch_size, num_nodes, num_nodes]
            attn_h = attention[:, :, :, h]
            
            # 提取当前头的节点特征 [batch_size, num_nodes, out_features]
            feat_h = Wh_reshaped[:, :, h, :]
            
            # 使用批量矩阵乘法计算加权和 [batch_size, num_nodes, out_features]
            output[:, :, h, :] = torch.bmm(attn_h, feat_h)
        
        # 变形回原始形状 [batch_size, num_nodes, num_heads * out_features]
        output = output.view(batch_size, num_nodes, self.num_heads * self.out_features)
        
        return output

class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, entity_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.entity_dim = entity_dim
        self.num_heads = num_heads
        
        # 修改: 不再要求entity_dim必须能被num_heads整除
        # 而是调整num_heads使其能够整除entity_dim
        if entity_dim % num_heads != 0:
            # 找到能够整除entity_dim的最大头数
            for i in range(num_heads-1, 0, -1):
                if entity_dim % i == 0:
                    self.num_heads = i
                    print(f"警告: 实体维度 {entity_dim} 不能被 {num_heads} 整除, 已调整为 {i} 个头")
                    break
            # 如果没有找到合适的头数，则使用1
            if entity_dim % self.num_heads != 0:
                self.num_heads = 1
                print(f"警告: 实体维度 {entity_dim} 不能被任何头数整除, 已调整为单头注意力")
        
        self.head_dim = entity_dim // self.num_heads
        
        self.query = nn.Linear(entity_dim, entity_dim)
        self.key = nn.Linear(entity_dim, entity_dim)
        self.value = nn.Linear(entity_dim, entity_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(entity_dim, entity_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
        self.attn = None
        
    def forward(self, x, mask=None):
        """
        x: [batch_size, num_nodes, entity_dim]
        mask: [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = x.size()
        
        # 线性变换
        q = self.query(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（如果有）
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attn = self.softmax(scores)
        attn = self.attn_dropout(attn)
        
        # 加权和
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.entity_dim)
        
        # 最后的线性变换
        output = self.proj(context)
        output = self.proj_dropout(output)
        
        return output, attn

class PositionwiseFeedForward(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, entity_dim, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(entity_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, entity_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class GraphTransformerLayer(nn.Module):
    """
    图Transformer层
    """
    def __init__(self, entity_dim, num_heads, d_ff, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        # 确保传入的头数能被实体维度整除
        if entity_dim % num_heads != 0:
            for i in range(num_heads-1, 0, -1):
                if entity_dim % i == 0:
                    num_heads = i
                    print(f"警告: GraphTransformerLayer - 实体维度 {entity_dim} 不能被 {num_heads} 整除, 已调整为 {i} 个头")
                    break
            if entity_dim % num_heads != 0:
                num_heads = 1
                print(f"警告: GraphTransformerLayer - 实体维度 {entity_dim} 不能被任何头数整除, 已调整为单头注意力")
        
        self.self_attn = MultiHeadSelfAttention(entity_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(entity_dim, d_ff, dropout)
        self.norm1 = nn.LayerNorm(entity_dim)
        self.norm2 = nn.LayerNorm(entity_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_mask=None):
        """
        x: [batch_size, num_nodes, entity_dim]
        adj_mask: [batch_size, num_nodes, num_nodes]
        """
        # 自注意力层
        attn_output, _ = self.self_attn(x, adj_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class MetaPathAggregator(nn.Module):
    """
    元路径聚合器，根据不同类型的边学习不同的元路径
    """
    def __init__(self, entity_dim, num_relation_types, num_heads=8):
        super(MetaPathAggregator, self).__init__()
        self.entity_dim = entity_dim
        self.num_relation_types = num_relation_types
        self.num_heads = num_heads
        
        # 确保头数能被实体维度整除
        if entity_dim % num_heads != 0:
            # 找到能够整除entity_dim的最大头数
            for i in range(num_heads-1, 0, -1):
                if entity_dim % i == 0:
                    self.num_heads = i
                    print(f"警告: MetaPathAggregator - 实体维度 {entity_dim} 不能被 {num_heads} 整除, 已调整为 {i} 个头")
                    break
            # 如果没有找到合适的头数，则使用1
            if entity_dim % self.num_heads != 0:
                self.num_heads = 1
                print(f"警告: MetaPathAggregator - 实体维度 {entity_dim} 不能被任何头数整除, 已调整为单头注意力")
        
        # 为不同类型的关系创建不同的注意力头
        self.relation_attentions = nn.ModuleList([
            GraphAttentionLayer(entity_dim, entity_dim // self.num_heads, self.num_heads)
            for _ in range(num_relation_types)
        ])
        
        # 最终的特征融合 - 修正: 使用固定的输入维度，不依赖于关系类型的数量
        # 因为可能只有部分关系类型被处理，所以不能简单地乘以num_relation_types
        self.fusion = nn.Linear(entity_dim, entity_dim)
        
    def forward(self, x, adj_list):
        """
        x: 节点特征 [batch_size, num_nodes, entity_dim]
        adj_list: 不同类型关系的邻接矩阵列表 [[batch_size, num_nodes, num_nodes], ...]
        """
        # 为每种关系类型计算表示
        relation_outputs = []
        for i, adj in enumerate(adj_list):
            if i < len(self.relation_attentions):
                relation_output = self.relation_attentions[i](x, adj)
                relation_outputs.append(relation_output)
        
        # 融合不同关系类型的表示 - 修正: 使用平均池化而不是拼接
        if relation_outputs:
            # 使用平均池化合并所有关系表示
            combined = torch.stack(relation_outputs, dim=0).mean(dim=0)
            output = F.relu(self.fusion(combined))
        else:
            output = F.relu(self.fusion(x))
            
        return output

class GraphTransformerNetwork(BaseGNNLayer):
    """
    基于论文《Graph Transformer Networks: Learning Meta-path Graphs to Improve GNNs》
    实现的图Transformer网络
    """
    def __init__(self, args, num_entity, num_relation, entity_dim):
        super(GraphTransformerNetwork, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.num_gtn_layers = args.get('num_gtn_layers', 2)
        
        # 修正：确保num_heads适合entity_dim
        self.num_heads = args.get('num_heads', 8)
        if entity_dim % self.num_heads != 0:
            # 找到能够整除entity_dim的最大头数
            for i in range(self.num_heads-1, 0, -1):
                if entity_dim % i == 0:
                    self.num_heads = i
                    print(f"警告: GraphTransformerNetwork - 实体维度 {entity_dim} 不能被 {args.get('num_heads', 8)} 整除, 已调整为 {i} 个头")
                    break
            # 如果没有找到合适的头数，则使用1
            if entity_dim % self.num_heads != 0:
                self.num_heads = 1
                print(f"警告: GraphTransformerNetwork - 实体维度 {entity_dim} 不能被任何头数整除, 已调整为单头注意力")
        
        # 如果d_ff为None，则设置为entity_dim的4倍
        if args.get('d_ff') is None:
            self.d_ff = 4 * entity_dim
        else:
            self.d_ff = args.get('d_ff')
            
        self.dropout = args.get('gtn_dropout', 0.1)
        
        # 初始化层
        self.init_layers(args)
        
    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        
        # 元路径聚合器
        self.meta_path_aggregator = MetaPathAggregator(
            entity_dim, 
            self.num_relation,
            self.num_heads
        )
        
        # 图Transformer层
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                entity_dim,
                self.num_heads,
                self.d_ff,
                self.dropout
            )
            for _ in range(self.num_gtn_layers)
        ])
        
        # 关系特征投影
        self.rel_projection = nn.Linear(entity_dim, entity_dim)
        
        # 节点更新层
        self.node_update = nn.Linear(2 * entity_dim, entity_dim)
        
    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, rel_features_inv, query_entities):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.rel_features_inv = rel_features_inv
        self.local_entity_emb = local_entity_emb
        self.possible_cand = []
        self.build_matrix()
        self.query_entities = query_entities
        
        # 构建不同关系类型的邻接矩阵
        self.build_relation_adj_matrices()
        
    def build_relation_adj_matrices(self):
        """
        为不同类型的关系构建邻接矩阵
        """
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        
        # 获取边列表中的数据
        batch_heads, batch_rels, batch_tails = self.edge_list[0], self.edge_list[1], self.edge_list[2]
        
        # 为每种关系类型创建邻接矩阵
        self.relation_adj_list = []
        
        # 创建所有关系的邻接矩阵
        adj_all = torch.zeros(batch_size, max_local_entity, max_local_entity).to(self.device)
        
        for i in range(len(batch_heads)):
            b, h, r, t = batch_heads[i] // max_local_entity, batch_heads[i] % max_local_entity, batch_rels[i], batch_tails[i] % max_local_entity
            adj_all[b, h, t] = 1
            
        self.relation_adj_list.append(adj_all)
        
        # 如果需要，可以为每种关系类型创建单独的邻接矩阵
        # 这里简化处理，只使用一个总的邻接矩阵
        
    def forward(self, current_dist, instruction, step=0, return_score=False):
        """
        GTN前向传播
        """
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        
        # 保存输入以用于残差连接
        h_v_l_minus_1 = self.local_entity_emb
        
        # 根据不同关系类型聚合节点表示
        meta_path_output = self.meta_path_aggregator(self.local_entity_emb, self.relation_adj_list)
        
        # 应用图Transformer层
        transformer_output = meta_path_output
        for layer in self.transformer_layers:
            transformer_output = layer(transformer_output, self.relation_adj_list[0])
        
        # 注入查询信息
        query_projected = instruction.mean(dim=1, keepdim=True).expand(-1, max_local_entity, -1)
        combined_features = torch.cat([transformer_output, query_projected], dim=-1)
        self.local_entity_emb = F.relu(self.node_update(combined_features))
        
        # 计算分数
        score_tp = self.score_func(self.local_entity_emb).squeeze(dim=-1)  # 确保维度正确
        
        # 应用掩码
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        
        # 计算概率分布
        current_dist = self.softmax_d1(score_tp)
        
        if return_score:
            return score_tp, current_dist
        
        return current_dist, self.local_entity_emb 