import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch


class MoleculeAtt(nn.Module):

    def __init__(self, embed_size, feature_size, hidden_size, device, dropout_rate=0.5):
        super(MoleculeAtt, self).__init__()
        self.embed_size = embed_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.device = device

        self.embedder = nn.Linear(self.feature_size, self.embed_size, bias=False)
        self.embedder.weight.data = torch.randn_like(self.embedder.weight)
        self.rnn = nn.LSTMCell(self.embed_size + self.hidden_size + self.feature_size, self.hidden_size, bias=True)
        self.final_projection = nn.Linear(self.hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_projection = nn.Linear(self.hidden_size, self.embed_size, bias=False)
        self.attention_projection.weight.data = torch.randn_like(self.attention_projection.weight)

    def forward(self, features: torch.Tensor, molecule_lengths: List[int]) -> torch.Tensor:
        embedding = self.embedder(features)
        mask = self.generate_mask(features, molecule_lengths)
        dec = self.step(embedding, mask, features)
        output = self.final_projection(self.dropout(torch.relu(dec)))
        return torch.squeeze(output).masked_fill_(mask.byte(), 0)

    def step(self, embedding: torch.Tensor, mask: torch.Tensor, features: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        q = torch.randn(embedding.size(0), self.hidden_size).to(self.device)
        c = torch.randn(embedding.size(0), self.hidden_size).to(self.device)
        r = torch.randn(embedding.size(0), self.embed_size).to(self.device)
        dec = []
        for i in range(embedding.size(1)):
            q_star = torch.cat([q, r, features[:, i, :].squeeze()], dim=1)
            q, c = self.rnn(q_star, (q, c))
            r = self.attention(q, embedding, mask)
            dec.append(q)
        return torch.stack(dec, dim=1).to(self.device)

    def generate_mask(self, features: torch.Tensor, molecule_lengths: List[int]) -> torch.Tensor:
        mask = torch.zeros(features.size(0), features.size(1), dtype=torch.float).to(self.device)
        for e_id, src_len in enumerate(molecule_lengths):
            mask[e_id, src_len:] = 1
        return mask

    def attention(self, q, embedding, mask):
        q_projection = self.attention_projection(q)
        e_t = torch.matmul(embedding, torch.unsqueeze(q_projection, dim=2)).squeeze()
        e_t.masked_fill_(mask.byte(), -float('inf'))
        alpha_t = F.softmax(e_t, 1).unsqueeze(2)
        r = torch.matmul(embedding.transpose(dim0=1, dim1=2), alpha_t).squeeze()
        return r
