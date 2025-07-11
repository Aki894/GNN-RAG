import torch
import numpy as np
from collections import defaultdict
import hashlib
import pickle

VERY_NEG_NUMBER = -100000000000

class BaseGNNLayer(torch.nn.Module):
    """
    Builds sparse tensors that represent structure.
    """
    # 稀疏矩阵缓存（类级别，所有实例共享）
    _sparse_matrix_cache = {}

    def __init__(self, args, num_entity, num_relation):
        super(BaseGNNLayer, self).__init__()
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.normalized_gnn = args['normalized_gnn']

    def _get_cache_key(self, edge_list, batch_size, max_local_entity, num_relation, normalized_gnn):
        # edge_list中每个元素都转为cpu numpy，再序列化
        edge_list_cpu = []
        for item in edge_list:
            if item is None:
                edge_list_cpu.append(None)
            elif isinstance(item, torch.Tensor):
                edge_list_cpu.append(item.detach().cpu().numpy())
            else:
                edge_list_cpu.append(np.array(item))
        # 用pickle序列化所有关键信息
        key_bytes = pickle.dumps((edge_list_cpu, batch_size, max_local_entity, num_relation, normalized_gnn))
        return hashlib.md5(key_bytes).hexdigest()

    def build_matrix(self):
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, _ = self.edge_list
        num_fact = len(fact_ids)
        num_relation = self.num_relation
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        self.num_fact = num_fact

        cache_key = self._get_cache_key(self.edge_list, batch_size, max_local_entity, num_relation, self.normalized_gnn)
        cache = BaseGNNLayer._sparse_matrix_cache
        if cache_key in cache:
            (
                self.fact2head_mat,
                self.head2fact_mat,
                self.fact2tail_mat,
                self.tail2fact_mat,
                self.head2tail_mat,
                self.fact2rel_mat,
                self.rel2fact_mat,
                self.batch_rels,
                self.batch_ids,
                self.batch_heads,
                self.batch_tails
            ) = cache[cache_key]
            return
        # 使用torch.stack来正确构建2D索引张量
        fact2head = torch.stack([batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.stack([batch_tails, fact_ids]).to(self.device)
        head2fact = torch.stack([fact_ids, batch_heads]).to(self.device)
        tail2fact = torch.stack([fact_ids, batch_tails]).to(self.device)
        head2tail = torch.stack([batch_heads, batch_tails]).to(self.device)
        rel2fact = torch.stack([fact_ids, batch_rels + batch_ids * num_relation]).to(self.device)
        fact2rel = torch.stack([batch_rels + batch_ids * num_relation, fact_ids]).to(self.device)
        self.batch_rels = torch.LongTensor(batch_rels).to(self.device)
        self.batch_ids = torch.LongTensor(batch_ids).to(self.device)
        self.batch_heads = torch.LongTensor(batch_heads).to(self.device)
        self.batch_tails = torch.LongTensor(batch_tails).to(self.device)
        if self.normalized_gnn:
            vals = torch.FloatTensor(weight_list).to(self.device)
        else:
            vals = torch.ones_like(self.batch_ids).float().to(self.device)
        self.fact2head_mat = self._build_sparse_tensor(fact2head, vals, (batch_size * max_local_entity, num_fact))
        self.head2fact_mat = self._build_sparse_tensor(head2fact, vals, (num_fact, batch_size * max_local_entity))
        self.fact2tail_mat = self._build_sparse_tensor(fact2tail, vals, (batch_size * max_local_entity, num_fact))
        self.tail2fact_mat = self._build_sparse_tensor(tail2fact, vals, (num_fact, batch_size * max_local_entity))
        self.head2tail_mat = self._build_sparse_tensor(head2tail, vals, (batch_size * max_local_entity, batch_size * max_local_entity))
        self.fact2rel_mat = self._build_sparse_tensor(fact2rel, vals, (batch_size * num_relation, num_fact))
        self.rel2fact_mat = self._build_sparse_tensor(rel2fact, vals, (num_fact, batch_size * num_relation))
        # 缓存
        cache[cache_key] = (
            self.fact2head_mat,
            self.head2fact_mat,
            self.fact2tail_mat,
            self.tail2fact_mat,
            self.head2tail_mat,
            self.fact2rel_mat,
            self.rel2fact_mat,
            self.batch_rels,
            self.batch_ids,
            self.batch_heads,
            self.batch_tails
        )

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)

    def build_adj_facts(self):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        max_fact = self.max_fact
        (e2f_batch, e2f_f, e2f_e, e2f_val), (f2e_batch, f2e_e, f2e_f, f2e_val) = self.edge_list2
        entity2fact_index = torch.LongTensor([e2f_batch, e2f_f, e2f_e]).to(self.device)
        entity2fact_val = torch.FloatTensor(e2f_val).to(self.device)
        self.entity2fact_mat =torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, \
            torch.Size([batch_size, max_fact, max_local_entity])).to(self.device) # batch_size, max_fact, max_local_entity
        fact2entity_index = torch.LongTensor([f2e_batch, f2e_e, f2e_f]).to(self.device)
        fact2entity_val = torch.FloatTensor(f2e_val).to(self.device)
        self.fact2entity_mat = torch.sparse.FloatTensor(fact2entity_index, fact2entity_val, \
            torch.Size([batch_size, max_local_entity, max_fact])).to(self.device) # batch_size,  max_local_entity, max_fact
        self.kb_fact_rel =  torch.LongTensor(self.kb_fact_rel).to(self.device)
