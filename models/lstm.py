import torch
import torch.nn as nn
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors


class LSTM(torch.nn.Module):

    def __init__(self, total_word, num_class, vectorizer, word2idx, idx2word,
                 embed_size = 300, hidden_size = 300,  padding_index=0, num_layers= 2, drouput_out = 0.2,
                 bool_initialize_weights = True, pretrained_wv_type = "word2vec"):
        '''

        :param total_word:
        :param num_class:
        :param vectorizer:
        :param word2idx:
        :param idx2word:
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
        self.bool_initialize_weights = bool_initialize_weights
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

        if bool_initialize_weights :
            # Initialize embedding matrix
            self.initialize_weights()
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
        self.embed.requires_grad_(False)

        return



    def initialize_weights(self):

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # Initialize weights with Xavier uniform distribution
                nn.init.xavier_uniform_(layer.weight)
                # Initialize biases to zero
                nn.init.zeros_(layer.bias)

    def load_sst2_weights(self, model_weights):
        """
           This function loads weights into the model for all layers except for the classifier.
           :param model_weights: A dictionary of pre-trained weights.
           """

        # Iterate through the state dictionary of the model
        current_model_state = self.state_dict()  # Get the current state of the model
        pretrained_state = model_weights  # Loaded pre-trained state dict

        # Filter out the classifier layers from the pretrained_state
        pretrained_state = {k: v for k, v in pretrained_state.items() if 'classifier' not in k}

        # Update the current model state with the pretrained weights
        current_model_state.update(pretrained_state)

        # Load the updated state dict into the model
        self.load_state_dict(current_model_state)

    def freeze_grads(self, freeze_classifier = False):
        self.embed.requires_grad_(False)
        self.lstm.requires_grad_(False)

        if freeze_classifier :
            self.classifier.requires_grad_(False)

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
        self.lstm.requires_grad_(False)