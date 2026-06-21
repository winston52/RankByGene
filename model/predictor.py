import torch.nn as nn


class GenePredictionMLP(nn.Module):
    """Downstream prediction head mapping learned image features to gene
    expression in the gene-prediction evaluation."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.1):
        super(GenePredictionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
