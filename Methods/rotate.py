"""
RotatE: Rotation-based Knowledge Graph Embedding
Reference: Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" (ICLR 2019)
"""

import numpy as np
import torch
import torch.nn as nn


class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=50, margin=6.0):
        super(RotatE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim * 2)  # real + imaginary
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.uniform_(self.relation_embeddings.weight, -np.pi, np.pi)

    def forward(self, heads, relations, tails):
        """Compute RotatE scoring in complex space"""
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        h_re, h_im = torch.chunk(h, 2, dim=1)
        t_re, t_im = torch.chunk(t, 2, dim=1)

        r_re = torch.cos(r)
        r_im = torch.sin(r)

        # Complex multiplication: h * r
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        score = torch.sqrt(torch.pow(hr_re - t_re, 2) + torch.pow(hr_im - t_im, 2)).sum(dim=1)
        return score

    def get_embeddings(self):
        return self.entity_embeddings.weight.data.cpu().numpy()
