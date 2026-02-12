"""
Graph Embedding Similarity Calculator
Provides unified interface for TransE and RotatE similarity calculation
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from .transe import TransE
from .rotate import RotatE


class GraphEmbeddingSimilarity:
    def __init__(self, embedding_dim=50):
        self.embedding_dim = embedding_dim
        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}

    def build_vocabulary(self, triples1, triples2):
        """Build entity and relation vocabularies from both graphs"""
        entities = set()
        relations = set()

        for triple in triples1 + triples2:
            if len(triple) == 3:
                entities.add(triple[0].lower())
                entities.add(triple[2].lower())
                relations.add(triple[1].lower())

        self.entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self.relation2id = {r: i for i, r in enumerate(sorted(relations))}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.id2relation = {i: r for r, i in self.relation2id.items()}

        return len(entities), len(relations)

    def triples_to_indices(self, triples):
        """Convert triples to index format"""
        indexed = []
        for triple in triples:
            if len(triple) == 3:
                h = self.entity2id.get(triple[0].lower(), 0)
                r = self.relation2id.get(triple[1].lower(), 0)
                t = self.entity2id.get(triple[2].lower(), 0)
                indexed.append((h, r, t))
        return indexed

    def train_model(self, model, triples, epochs=100, lr=0.01):
        """Train embedding model with margin ranking loss"""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            for h, r, t in triples:
                heads = torch.LongTensor([h])
                relations = torch.LongTensor([r])
                tails = torch.LongTensor([t])

                pos_score = model(heads, relations, tails)

                # Negative sampling
                neg_h = np.random.randint(0, len(self.entity2id))
                neg_t = np.random.randint(0, len(self.entity2id))

                neg_heads = torch.LongTensor([neg_h])
                neg_tails = torch.LongTensor([neg_t])

                neg_score = model(neg_heads, relations, neg_tails)

                loss = torch.clamp(model.margin + pos_score - neg_score, min=0).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def get_graph_embedding(self, model, triples):
        """Get graph-level embedding by averaging entity embeddings"""
        entity_embeddings = model.get_embeddings()

        entities = set()
        for triple in triples:
            entities.add(triple[0])
            entities.add(triple[2])

        embeddings = []
        for entity in entities:
            entity_id = self.entity2id.get(entity.lower(), 0)
            embeddings.append(entity_embeddings[entity_id])

        if len(embeddings) == 0:
            embedding_dim = self.embedding_dim * 2 if isinstance(model, RotatE) else self.embedding_dim
            return np.zeros(embedding_dim)

        return np.mean(embeddings, axis=0)

    def calculate_transe_similarity(self, triples1, triples2):
        """Calculate similarity using TransE embeddings"""
        num_entities, num_relations = self.build_vocabulary(triples1, triples2)

        if num_entities == 0 or num_relations == 0:
            return None

        indexed1 = self.triples_to_indices(triples1)
        indexed2 = self.triples_to_indices(triples2)

        if len(indexed1) == 0 or len(indexed2) == 0:
            return None

        model = TransE(num_entities, num_relations, self.embedding_dim)
        all_triples = indexed1 + indexed2
        self.train_model(model, all_triples, epochs=100)

        emb1 = self.get_graph_embedding(model, triples1)
        emb2 = self.get_graph_embedding(model, triples2)

        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)

    def calculate_rotate_similarity(self, triples1, triples2):
        """Calculate similarity using RotatE embeddings"""
        num_entities, num_relations = self.build_vocabulary(triples1, triples2)

        if num_entities == 0 or num_relations == 0:
            return None

        indexed1 = self.triples_to_indices(triples1)
        indexed2 = self.triples_to_indices(triples2)

        if len(indexed1) == 0 or len(indexed2) == 0:
            return None

        model = RotatE(num_entities, num_relations, self.embedding_dim)
        all_triples = indexed1 + indexed2
        self.train_model(model, all_triples, epochs=100)

        emb1 = self.get_graph_embedding(model, triples1)
        emb2 = self.get_graph_embedding(model, triples2)

        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
