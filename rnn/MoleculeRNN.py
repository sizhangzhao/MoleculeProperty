import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch


class MoleculeRNN(nn.Module):

    def __init__(self, feature_size, hidden_size, dropout_rate=0.5,
                 use_attention=True, device="cpu"):
        super(MoleculeRNN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size

        self.bi_lstm = nn.LSTM(feature_size, hidden_size, bidirectional=True, batch_first=True)
        self.combined_output_projection = nn.Linear((2 * hidden_size + feature_size), hidden_size, bias=True)
        self.final_projection = nn.Linear(hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_attention = use_attention
        self.device = device
        if self.use_attention:
            self.method = "quadratic"
            if self.method == "neural":
                self.attention_projection = nn.Linear((2 * feature_size), 1, bias=True)
            elif self.method == "quadratic":
                self.attention_weight = self.new_parameter(self.feature_size, self.feature_size).to(self.device)

    def forward(self, features: torch.Tensor, molecule_lengths: List[int]) -> torch.Tensor:
        X = nn.utils.rnn.pack_padded_sequence(features, molecule_lengths, batch_first=True)
        enc_hiddens, (_, _) = self.bi_lstm(X)
        enc_hiddens = nn.utils.rnn.pad_packed_sequence(enc_hiddens, batch_first=True)[0]
        mask = self.generate_sent_mask(enc_hiddens, molecule_lengths)
        if self.use_attention:
            enc_hiddens = self.attention(features, enc_hiddens, mask, self.method)
        output_feature = torch.cat([enc_hiddens, features], 2)
        projection = self.combined_output_projection(output_feature)
        output = self.final_projection(self.dropout(torch.relu(projection)))
        return torch.squeeze(output).masked_fill_(mask.byte(), 0)

    def generate_sent_mask(self, enc_hiddens: torch.Tensor, molecule_lengths) -> torch.Tensor:
        enc_mask = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(molecule_lengths):
            enc_mask[e_id, src_len:] = 1
        return enc_mask.to(self.device)

    def attention(self, features: torch.Tensor, enc_hiddens: torch.Tensor, mask, method="neural") -> torch.Tensor:
        if method == "neural":
            return self.attention_neural(features, enc_hiddens, mask)
        if method == "quadratic":
            return self.attention_quadratic(features, enc_hiddens, mask)

    def attention_neural(self, features: torch.Tensor, enc_hiddens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        src_len = features.size(1)
        attention_scores = []
        for feature_index in range(src_len):
            current_feature = torch.repeat_interleave(torch.unsqueeze(features[:, feature_index, :], dim=1),
                                                      src_len, dim=1)
            attention_feature = torch.cat([features, current_feature], 2)
            attention_score = self.dropout(torch.relu(self.attention_projection(attention_feature)))
            attention_scores.append(attention_score)
        attention_scores = torch.squeeze(torch.stack(attention_scores, 2))
        attention_scores.masked_fill_(torch.unsqueeze(mask, 2).byte(), -float('inf'))
        attention_score_softmax = F.softmax(attention_scores, 1)
        enc_hiddens = torch.matmul(attention_score_softmax, enc_hiddens)
        return enc_hiddens

    def attention_quadratic(self, features: torch.Tensor, enc_hiddens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention_weight = self.attention_weight
        attention_score = torch.matmul(features, attention_weight)
        attention_score = torch.matmul(attention_score, features.transpose(1, 2)).squeeze()
        attention_score.masked_fill_(torch.unsqueeze(mask, 2).byte(), -float('inf'))
        attention_score_softmax = F.softmax(attention_score, 1)
        enc_hiddens = torch.matmul(attention_score_softmax, enc_hiddens)
        return enc_hiddens

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size).cuda()) #TODO: has to go this way otherwise it wont be recognized as parameter
        torch.nn.init.xavier_normal_(out)
        return out
