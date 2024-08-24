import torch
import torch.nn as nn
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors


class LSTM(torch.nn.Module):

    def __init__(self, total_word, num_class, vectorizer, word2idx, idx2word, device,
                 embed_size = 300, hidden_size = 300,  padding_index=0, num_layers= 5, drouput_out = 0.2,
                 initialize_weights = True, pretrained_wv_type = "glove"):
        '''

        :param total_word:
        :param num_class:
        :param vectorizer:
        :param word2idx:
        :param idx2word:
        :param device:
        :param embed_size:
        :param hidden_size:
        :param padding_index:
        :param num_layers:
        :param drouput_out:
        :param initialize_weights:
        :param pretrained_wv_type:  Options word2vec, glove
        '''

        super().__init__()
        self.total_word = total_word # Total words present in vocabulary
        self.embed_size = embed_size # Embedding size Generally it is around 300
        self.hidden_size = hidden_size # Hidden size
        self.num_class = num_class # Number of classes to predict
        self.vectorizer = vectorizer
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.device = device
        self.initialize_weights = initialize_weights
        self.pretrained_wv_type = pretrained_wv_type

        self.embed = torch.nn.Embedding(num_embeddings=total_word,
                                        embedding_dim=embed_size,
                                        padding_idx=padding_index)


        self.lstm = torch.nn.LSTM(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  num_layers= num_layers,
                                  bidirectional=True,
                                  batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=2 * hidden_size, out_features=4 * hidden_size),
            nn.LeakyReLU(negative_slope=0.01),  # LeakyReLU activation
            nn.Dropout(p=drouput_out),  # Dropout layer

            nn.Linear(in_features=4 * hidden_size, out_features=hidden_size),
            nn.LeakyReLU(negative_slope=0.01),  # LeakyReLU activation
            nn.Dropout(p=drouput_out),  # Dropout layer

            nn.Linear(in_features=hidden_size, out_features=num_class)
        )

        # Initialize embedding matrix
        self.initialize_embedding_matrix()

    def forward(self, X):
        out = self.embed(X)
        out, _ = self.lstm(out)
        out = self.classifier(out[:, -1, :])
        return out

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
        return