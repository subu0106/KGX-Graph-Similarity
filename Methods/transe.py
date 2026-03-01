"""
TransE: Translation-based Knowledge Graph Embedding
Reference: Bordes et al. "Translating Embeddings for Modeling Multi-relational Data" (NIPS 2013)
"""

import torch
import torch.nn as nn


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=50, margin=1.0):
        super(TransE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, relations, tails):
        """Compute TransE scoring: ||h + r - t||"""
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        score = torch.norm(h + r - t, p=2, dim=1)
        return score

    def get_embeddings(self):
        return self.entity_embeddings.weight.data.cpu().numpy()
