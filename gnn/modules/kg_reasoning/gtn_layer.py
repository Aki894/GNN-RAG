import torch
import torch.nn as nn
import torch.nn.functional as F
# 移除对torch_sparse的依赖
# import torch_sparse

from .base_gnn import BaseGNNLayer

class GTNLayer(BaseGNNLayer):
    """
    问题感知的图转换网络层 (Question-Aware Graph Transformer Network Layer)
    
    该层实现了动态的元路径学习，能够根据问题指令生成适合当前问题的元路径图。
    """
    def __init__(self, args, num_entity, num_relation, entity_dim, num_channels=1):
        super(GTNLayer, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.num_channels = num_channels  # 元路径通道数
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        
        # 问题感知的权重生成网络
        self.question_weight_net = nn.Sequential(
            nn.Linear(entity_dim, entity_dim),
            nn.ReLU(),
            nn.Linear(entity_dim, num_relation * num_channels)
        )
        
        # 元路径组合层
        self.meta_path_layer1 = nn.Linear(entity_dim, entity_dim)
        self.meta_path_layer2 = nn.Linear(entity_dim, entity_dim)
        
        # 门控融合机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(entity_dim * 2, entity_dim),
            nn.Sigmoid()
        )
        
        # 输出层
        self.output_transform = nn.Linear(entity_dim, entity_dim)
        
    def init_local_entity(self, local_entity):
        """
        初始化local_entity属性
        
        Args:
            local_entity: 局部实体映射张量
        """
        self.local_entity = local_entity
        
    def init_edge_list(self, edge_list, batch_size, max_local_entity):
        """
        初始化边列表信息
        
        Args:
            edge_list: 边列表元组
            batch_size: 批次大小
            max_local_entity: 最大局部实体数
        """
        self.edge_list = edge_list
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.build_matrix()
        
    def forward(self, local_entity_emb, question_emb, curr_dist):
        """
        根据问题指令和当前实体概率分布，生成元路径图并进行消息传递
        
        Args:
            local_entity_emb: 实体表示 [batch_size, max_local_entity, entity_dim]
            question_emb: 问题指令表示 [batch_size, entity_dim]
            curr_dist: 当前实体概率分布 [batch_size, max_local_entity]
            
        Returns:
            meta_path_entity_emb: 元路径图上的实体表示
            meta_path_adj: 生成的元路径图邻接矩阵
        """
        batch_size, max_local_entity = local_entity_emb.size(0), local_entity_emb.size(1)
        
        # 1. 根据问题指令动态生成关系权重
        relation_weights = self.question_weight_net(question_emb)  # [batch_size, num_relation * num_channels]
        relation_weights = relation_weights.view(batch_size, self.num_channels, self.num_relation)
        relation_weights = F.softmax(relation_weights, dim=-1)  # [batch_size, num_channels, num_relation]
        
        # 2. 构建原始图的邻接矩阵列表
        # self.edge_list已经在init_edge_list中设置
        # self.build_matrix()已经在init_edge_list中调用
        
        # 3. 为每个通道生成元路径图
        meta_path_entity_embs = []
        
        for c in range(self.num_channels):
            # 3.1 获取当前通道的关系权重
            channel_weights = relation_weights[:, c, :]  # [batch_size, num_relation]
            
            # 3.2 对每个批次单独处理
            channel_entity_embs = []
            for b in range(batch_size):
                # 获取当前批次的权重
                weights = channel_weights[b]  # [num_relation]
                
                # 获取当前批次的实体表示
                entity_emb = local_entity_emb[b]  # [max_local_entity, entity_dim]
                
                # 应用元路径变换
                transformed_emb = self.meta_path_layer1(entity_emb)  # [max_local_entity, entity_dim]
                
                # 初始化邻居表示
                neighbor_emb = torch.zeros_like(transformed_emb).to(self.device)
                
                # 根据权重组合关系进行消息传递
                for r in range(self.num_relation):
                    # 获取关系r的边
                    rel_mask = (self.batch_rels == r) & (self.batch_ids == b)
                    if not rel_mask.any():
                        continue
                    
                    # 获取当前关系下的头尾实体
                    rel_heads = self.batch_heads[rel_mask]
                    rel_tails = self.batch_tails[rel_mask]
                    
                    # 当前关系的权重
                    rel_weight = weights[r]
                    
                    # 对每条边进行消息传递
                    for h, t in zip(rel_heads, rel_tails):
                        # 从头实体到尾实体传递消息
                        # 确保h和t是标量整数，避免"only integer scalar arrays"错误
                        local_h_indices = (self.local_entity[b] == h.item()).nonzero(as_tuple=True)
                        if local_h_indices[0].numel() == 0:  # 头实体不在局部图中
                            continue
                        h_local = local_h_indices[0].item() # 这会得到全局实体在局部张量中的索引，例如 0-1999

                        local_t_indices = (self.local_entity[b] == t.item()).nonzero(as_tuple=True)
                        if local_t_indices[0].numel() == 0:  # 尾实体不在局部图中
                            continue
                        t_local = local_t_indices[0].item() # 这会得到全局实体在局部张量中的索引，例如 0-1999

                        neighbor_emb[t_local] += transformed_emb[h_local] * rel_weight
                
                # 应用第二层变换
                neighbor_emb = self.meta_path_layer2(neighbor_emb)
                channel_entity_embs.append(neighbor_emb)
            
            # 将批次结果堆叠
            channel_entity_emb = torch.stack(channel_entity_embs)  # [batch_size, max_local_entity, entity_dim]
            meta_path_entity_embs.append(channel_entity_emb)
            
        # 4. 聚合所有通道的结果
        if len(meta_path_entity_embs) > 1:
            meta_path_entity_emb = torch.mean(torch.stack(meta_path_entity_embs), dim=0)
        else:
            meta_path_entity_emb = meta_path_entity_embs[0]
        
        return meta_path_entity_emb
    
    def fusion_with_original(self, original_emb, meta_path_emb):
        """
        将原始图和元路径图的实体表示进行门控融合
        
        Args:
            original_emb: 原始图上的实体表示
            meta_path_emb: 元路径图上的实体表示
            
        Returns:
            fused_emb: 融合后的实体表示
        """
        # 计算融合门控值
        concat_emb = torch.cat([original_emb, meta_path_emb], dim=-1)
        gate = self.fusion_gate(concat_emb)
        
        # 门控融合
        fused_emb = gate * meta_path_emb + (1 - gate) * original_emb
        
        # 最终变换
        fused_emb = self.output_transform(fused_emb)
        
        return fused_emb 