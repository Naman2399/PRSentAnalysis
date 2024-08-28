import torch
import torch.nn as nn
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
import torch.nn.functional as F
class CNN2(torch.nn.Module):

    def __init__(self, total_word, num_class, vectorizer, word2idx, idx2word,
                 embed_size=300, hidden_size=300, padding_index=0, dropout_out=0.5,  # Increased dropout
                 bool_initialize_weights=True, pretrained_wv_type="word2vec"):
        super().__init__()
        self.total_word = total_word
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.vectorizer = vectorizer
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.bool_initialize_weights = bool_initialize_weights
        self.pretrained_wv_type = pretrained_wv_type

        self.embed = torch.nn.Embedding(num_embeddings=total_word,
                                        embedding_dim=embed_size,
                                        padding_idx=padding_index)

        self.conv_block_1 = nn.Sequential(
            nn.Dropout(0.5),  # Increased dropout rate
            nn.Conv1d(in_channels=embed_size, out_channels=max(embed_size // 4, 1), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),  # Added MaxPooling to reduce dimensionality
            nn.Dropout(0.5),  # Increased dropout rate
            nn.Conv1d(in_channels=max(embed_size // 4, 1), out_channels=max(embed_size // 8, 1), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),  # Added MaxPooling
            nn.Dropout(0.5)  # Increased dropout rate
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(max(embed_size // 8, 1), 128),
            nn.Dropout(dropout_out),
            nn.LeakyReLU(),
            nn.Linear(128, num_class)
        )

        if bool_initialize_weights:
            self.initialize_weights()
            self.initialize_embedding_matrix()

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = self.conv_block_1(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

    # Your existing methods...

# Testing the modified model
if __name__ == "__main__" :
    B = 32  # Batch size
    seq_len = 10  # Sequence length
    random_tensor = torch.randint(low=1, high=501, size=(B, seq_len))
    model = CNN2(total_word=1000, num_class=5, vectorizer=None, idx2word=None, word2idx=None, bool_initialize_weights=False)
    output = model(random_tensor)
