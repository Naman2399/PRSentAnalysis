import torch

class SentimentAnalysis(torch.nn.Module):

    def __init__(self, total_word, embed_size, hidden_size, num_class, padding_index=0):
        super().__init__()
        self.total_word = total_word
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_class = num_class

        self.embed = torch.nn.Embedding(num_embeddings=total_word,
                                        embedding_dim=embed_size,
                                        padding_idx=padding_index)
        self.lstm = torch.nn.LSTM(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=1,
                                  bidirectional=True,
                                  batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=2 * self.hidden_size, out_features=num_class)
        )

    def forward(self, X):
        out = self.embed(X)
        out, _ = self.lstm(out)
        out = self.classifier(out[:, -1, :])
        return out