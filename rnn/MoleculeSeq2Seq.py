import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch


class MoleculeSeq2Seq(nn.Module):

    def __init__(self, embed_size, feature_size, hidden_size, device, dropout_rate=0.5):
        super(MoleculeSeq2Seq, self).__init__()
        self.embed_size = embed_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.device = device

        self.embedder = nn.Linear(self.feature_size, self.embed_size, bias=False)
        self.embedder.weight.data = torch.randn_like(self.embedder.weight)
        self.encoder = nn.LSTMCell(self.embed_size + self.hidden_size, self.hidden_size, bias=True)
        self.decoder = nn.LSTMCell(self.feature_size, self.hidden_size, bias=True)
        self.final_projection = nn.Linear(self.hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_projection = nn.Linear(self.hidden_size, self.embed_size, bias=False)
        self.attention_projection.weight.data = torch.randn_like(self.attention_projection.weight)

    def forward(self, features: torch.Tensor, molecule_lengths: List[int]) -> torch.Tensor:
        embedding = self.embedder(features)
        mask = self.generate_mask(features, molecule_lengths)
        enc_h, enc_c = self.encode(embedding, mask)
        dec = self.decode(features, enc_h, enc_c)
        output = self.final_projection(self.dropout(torch.relu(dec)))
        return torch.squeeze(output).masked_fill_(mask.byte(), 0)

    def encode(self, embedding: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        q = torch.randn(embedding.size(0), self.hidden_size).to(self.device)
        c = torch.randn(embedding.size(0), self.hidden_size).to(self.device)
        r = torch.randn(embedding.size(0), self.embed_size).to(self.device)
        for i in range(embedding.size(1)):
            q_star = torch.cat([q, r], dim=1)
            q, c = self.encoder(q_star, (q, c))
            r = self.attention(q, embedding, mask)
        return q, c

    def decode(self, features: torch.Tensor, enc_h: torch.Tensor, enc_c: torch.Tensor) -> torch.Tensor:
        dec = torch.zeros(features.size(0), features.size(1), self.hidden_size, dtype=torch.float).to(self.device)
        for pos in range(features.size(1)):
            dec[:, pos, :] = self.decoder(features[:, pos, :], (enc_h, enc_c))[0]
        return dec

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
