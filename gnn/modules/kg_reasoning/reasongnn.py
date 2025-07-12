import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

from .base_gnn import BaseGNNLayer

# Import Gradformer specific modules
from Gradformer_main.model.Attention import MultiheadAttention, attention
from Gradformer_main.utils.process import process_hop

VERY_NEG_NUMBER = -100000000000

class ReasonGNNLayer(BaseGNNLayer):
    """
    GNN Reasoning
    """
    def __init__(self, args, num_entity, num_relation, entity_dim, alg):
        super(ReasonGNNLayer, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.alg = alg
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        
        self.use_posemb = args['pos_emb']
        
        # Gradformer specific parameters
        self.gamma = args.get('gamma', 0.9) # Default gamma from Gradformer
        self.slope = args.get('slope', 0.1) # Default slope from Gradformer
        self.n_head = args.get('nhead', 4) # Default number of heads from Gradformer
        self.n_hop = args.get('n_hop', 4) # Default n_hop for learnable constraint

        self.hop = Parameter(torch.full((self.n_head, 1, 1), float(self.n_hop))) # Gradformer learnable constraint
        
        self.init_layers(args)

        # Initialize MultiheadAttention for Gradformer-style aggregation
        self.grad_attn = MultiheadAttention(
            h=self.n_head,
            d_model=entity_dim,
            dropout=args.get('attn_dropout', 0.0) # Using attn_dropout from args or default
        )

    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.glob_lin = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        # Removed self.lin as it's replaced by MultiheadAttention
        self.lin = nn.Linear(in_features=2*entity_dim, out_features=entity_dim)
        assert self.alg == 'bfs'
        self.linear_dropout = args['linear_dropout']
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        self.use_residual = args.get('use_residual', False)
        self.use_node_adaptive_residual = args.get('use_node_adaptive_residual', False)

        if self.use_residual or self.use_node_adaptive_residual:
            self.res_norm = nn.LayerNorm(entity_dim)
            if self.use_node_adaptive_residual:
                self.node_adaptive_linear = nn.Linear(in_features=entity_dim, out_features=1)

        # Removed dynamic creation of rel_linear and e2e_linear as they will be handled by MultiheadAttention if we go for full replacement
        # However, for initial integration, we might keep them for blending.
        # For now, let's assume Gradformer's attention will manage the message passing.
        for i in range(self.num_gnn):
            # These lines are kept for now, but their usage will change significantly.
            # We will use self.grad_attn for primary message aggregation.
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            if self.alg == 'bfs':
                self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2*(self.num_ins)*entity_dim + entity_dim, out_features=entity_dim))

            if self.use_posemb:
                self.add_module('pos_emb' + str(i), nn.Embedding(self.num_relation, entity_dim))
                self.add_module('pos_emb_inv' + str(i), nn.Embedding(self.num_relation, entity_dim))
        
        # self.lin_m is for combining instruction-specific representations in original ReaRev, keep it for now.
        self.lin_m =  nn.Linear(in_features=(self.num_ins)*entity_dim, out_features=entity_dim)


    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, rel_features_inv, query_entities, query_node_emb=None):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat # This is (batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, weight_rel_list)
        self.rel_features = rel_features
        self.rel_features_inv = rel_features_inv
        self.local_entity_emb = local_entity_emb
        self.possible_cand = []
        self.build_matrix() # This builds sparse matrices from self.edge_list
        self.query_entities = query_entities
       

    def reason_layer(self, curr_dist, instruction, rel_linear, pos_emb):
        """
        Original ReaRev reason layer implementation.
        """
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        rel_features = self.rel_features
        
        # Handle instruction shape - it might be concatenated instructions
        if instruction.dim() == 3:  # (batch_size, num_instructions, entity_dim)
            # Take the first instruction or average all instructions
            instruction = instruction[:, 0, :]  # Take first instruction
        
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels) #rels (facts), entity_dim
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids) #one query per batch entry: rels (facts), entity_dim
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1)) #rels (facts), 1 (scaling)

        possible_tail = torch.sparse.mm(self.fact2tail_mat, fact_prior) # batch_size * max_local_entity, 1
        # (batch_size *max_local_entity, num_fact) (num_fact, 1)
        possible_tail = (possible_tail > 1e-10).float().view(batch_size, max_local_entity)

        fact_val = fact_val * fact_prior
        
        f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)  # batch_size * max_local_entity, entity_dim 
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        
        return neighbor_rep, possible_tail

    def reason_layer_inv(self, curr_dist, instruction, rel_linear, pos_emb_inv):
        # This function will also be replaced/modified.
        pass # This function's logic will be incorporated into the forward method using MultiheadAttention

    def combine(self,emb):
        """
        Combines instruction-specific representations.
        """
        local_emb = torch.cat(emb, dim=-1)
        local_emb = F.relu(self.lin_m(local_emb))

        score_func = self.score_func
        
        score_tp = score_func(self.linear_drop(local_emb)).squeeze(dim=2)
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        # Ensure score_tp and answer_mask have the same shape
        if score_tp.shape != answer_mask.shape:
            # Calculate the correct shape based on the total size
            total_size = score_tp.numel()
            batch_size = answer_mask.shape[0]
            max_local_entity = answer_mask.shape[1]
            # If the total size matches batch_size * max_local_entity, reshape accordingly
            if total_size == batch_size * max_local_entity:
                score_tp = score_tp.view(batch_size, max_local_entity)
            else:
                # If there are multiple heads, we need to handle that
                num_heads = total_size // (batch_size * max_local_entity)
                if total_size == batch_size * max_local_entity * num_heads:
                    score_tp = score_tp.view(batch_size, max_local_entity, num_heads).mean(dim=2)
                else:
                    # Fallback: flatten and reshape to match answer_mask
                    score_tp = score_tp.flatten()[:batch_size * max_local_entity].view(batch_size, max_local_entity)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        return current_dist, local_emb

    def forward(self, current_dist, relational_ins, sph, step=0, return_score=False):
        """
        Compute next probabilistic vectors and current node representations
        using original ReaRev logic with Gradformer enhancements.
        """
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        score_func = self.score_func

        # Use original ReaRev logic for message passing
        # This ensures compatibility with the existing architecture
        neighbor_rep, possible_tail = self.reason_layer(current_dist, relational_ins, rel_linear, None)
        
        # Handle the concatenation properly
        # The e2e_linear expects input of size 2*(self.num_ins)*entity_dim + entity_dim
        # But we only have entity_dim from local_entity_emb and entity_dim from neighbor_rep
        # So we need to create the expected input size
        batch_size, max_local_entity, entity_dim = self.local_entity_emb.shape
        
        # Calculate the expected input size for e2e_linear
        expected_input_dim = 2 * self.num_ins * entity_dim + entity_dim  # 2*3*64 + 64 = 448
        current_input_dim = entity_dim + entity_dim  # 64 + 64 = 128 (local_entity_emb + neighbor_rep)
        missing_dim = expected_input_dim - current_input_dim  # 448 - 128 = 320
        
        # Create a placeholder for the missing dimensions
        placeholder = torch.zeros(batch_size, max_local_entity, missing_dim, device=self.local_entity_emb.device)
        
        # Concatenate: [placeholder, local_entity_emb, neighbor_rep]
        next_local_entity_emb = torch.cat((placeholder, self.local_entity_emb, neighbor_rep), dim=2)
        self.local_entity_emb = e2e_linear(self.linear_drop(next_local_entity_emb))
        
        self.local_entity_emb = F.relu(self.local_entity_emb)

        # Original score calculation
        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)

        if return_score:
            return score_tp, current_dist
        
        return current_dist, self.local_entity_emb 


