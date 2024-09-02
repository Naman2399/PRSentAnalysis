import gensim.downloader as api
import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors


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

    def initialize_embedding_matrix(self):

        mask = np.ones((self.total_word, self.embed_size))
        embedding_matrix_weights = np.zeros((self.total_word, self.embed_size))

        if self.pretrained_wv_type == "word2vec" :
            # Word2Vec embeddings
            word2vec_model = api.load('word2vec-google-news-300')

            for i in range(self.total_word) :
                if i in self.idx2word.keys() :
                    try :
                        vector = word2vec_model[self.idx2word[i]]
                        embedding_matrix_weights[i] = vector
                        mask[i, :] = 0
                    except :
                        continue

        elif self.pretrained_wv_type == "glove" :
            glove_file_path = "/mnt/hdd/karmpatel/naman/demo/glove/glove.42B.300d.txt"
            glove_model = KeyedVectors.load_word2vec_format(glove_file_path, no_header=True, binary=False)

            for i in range(self.total_word) :
                if i in self.idx2word.keys() :
                    word = self.idx2word[i]
                    vector = glove_model.get_vector(word) if word in glove_model else None
                    if vector is not None:
                        embedding_matrix_weights[i] = vector
                        mask[i, :] = 0
                    else :
                        continue

        embedding_matrix_weights = torch.tensor(embedding_matrix_weights, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        updated_weights = self.embed.weight * mask + embedding_matrix_weights
        self.embed.from_pretrained(updated_weights)
        return

    def initialize_weights(self):

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # Initialize weights with Xavier uniform distribution
                nn.init.xavier_uniform_(layer.weight)
                # Initialize biases to zero
                nn.init.zeros_(layer.bias)

    def load_weights_except_classifier(self, checkpoint):
        """
        Loads the model weights from a checkpoint file except for the classifier layers.

        Args:
            checkpoint_path (str): Path to the checkpoint file containing the saved weights.
        """

        # Get the current state_dict of the model
        model_dict = self.state_dict()

        # Filter out classifier weights from the checkpoint
        pretrained_dict = {k: v for k, v in checkpoint.items() if "classifier" not in k and "embed" not in k}

        # Update the current state_dict with pretrained_dict, excluding classifier
        model_dict.update(pretrained_dict)

        # Load the updated state_dict back into the model
        self.load_state_dict(model_dict)

        print("Loaded pretrained weights, except classifier.")

    def freeze_weights(self):
        self.conv_block_1.requires_grad_(False)
        self.pool.requires_grad_(False)


# Testing the modified model
if __name__ == "__main__" :
    B = 32  # Batch size
    seq_len = 10  # Sequence length
    random_tensor = torch.randint(low=1, high=501, size=(B, seq_len))
    model = CNN2(total_word=1000, num_class=5, vectorizer=None, idx2word=None, word2idx=None, bool_initialize_weights=False)
    output = model(random_tensor)
