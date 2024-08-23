import os
import string

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def data_visulaization(df : pd.DataFrame) :

    # Data visualization
    sns.countplot(data=df, x='Rating', palette='mako').set_title('Rating Distribution Across Dataset')
    plt.savefig('/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/plots/rating_distribution.svg', format='svg')

    # Length of word in sentence
    df['Length'] = df['Review'].apply(lambda r: len(str(r).split(" ")))
    df.head()

    # Data visualization
    sns.displot(data=df, x='Length', hue='Rating', palette='mako', kind='kde', fill=True, aspect=4)
    plt.savefig('/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/plots/sentence_length_distribution.svg', format='svg')

    # Data visualization
    sns.FacetGrid(data=df, col='Rating').map(plt.hist, 'Length', color='#1D3557')
    plt.savefig('/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/plots/sentence_length_histogram.svg', format='svg')

def rating(score) :
    return int(score)

def clean_text(text) :
    # remove punctuations and uppercase
    text = str(text)
    clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower()

    # remove stopwords
    clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]

    # lemmatize the word
    sentence = []
    for word in clean_text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))

    return ' '.join(sentence)



def text_preprocessing(df : pd.DataFrame) :

    # Preprocessing rating
    df['Rating'] = df['Rating'].apply(rating)
    df['Rating'] = df['Rating'] - df['Rating'].min()
    # Cleaning Review Text
    df['Review'] = df['Review'].apply(clean_text)
    # Length of Dataset
    df['Length'] = df['Review'].apply(lambda r: len(r.split(" ")))
    new_length = df['Length'].sum()
    print('Total word after cleaning: {}'.format(new_length))
    return df

def dataframe_to_csv(df : pd.DataFrame, dir_path : str, file_name : str) :
    file_path = os.path.join(dir_path, file_name )
    df.to_csv(file_path)
    return file_path

def preprocessing(df : pd.DataFrame, dir_path = "../data", file_name = None ) :

    if file_name is None :
        print("Enter a valid file name")
        exit()

    data_visulaization(df)
    # Preprocessing dataset
    df = text_preprocessing(df)
    processed_file_path = dataframe_to_csv(df, dir_path, file_name)
    return processed_file_path

def split_train_test_validation(df : pd.DataFrame, valid_test_ratio =0.2 , valid_ratio_from_test = 0.5) :

    df['Review'] = df['Review'].astype(str)
    df['Rating'] = df['Rating'].astype(int)

    # Step 1: Split into training and temporary sets (temporary = test + validation)
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['Review'], df['Rating'], test_size= valid_test_ratio, stratify=df['Rating'], random_state=42
    )

    # Step 2: Split the temporary set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size= valid_ratio_from_test, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data_loaders(file_path, valid_test_ration =0.2, valid_ratio_from_test =0.5,
                     output_seq_len = 80, batch_size = 64) :

    # Using Post processed csv file
    print(f"Reading data from : {file_path} ")
    df = pd.read_csv(file_path)
    # Get details
    unique_ratings = df['Rating'].unique()
    num_classes = len(unique_ratings)
    print(f"Number of classes : {num_classes}")
    rating_counts = df['Rating'].value_counts()
    rating_counts_sorted = rating_counts.sort_index()

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_validation(df)

    # Check the size sets
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Building Vocabulary, convert text -> number
    vectorizer = TextVectorization(
        max_tokens=None,
        output_mode="int",
        output_sequence_length=output_seq_len
    )
    vectorizer.adapt(X_train)

    X_train = vectorizer(X_train)
    X_val = vectorizer(X_val)
    X_test = vectorizer(X_test)

    # Convert them to tensor
    X_train_tf = tf.convert_to_tensor(X_train)
    X_val_tf = tf.convert_to_tensor(X_val)
    X_test_tf = tf.convert_to_tensor(X_test)
    y_train_tf = tf.convert_to_tensor(y_train)
    y_val_tf = tf.convert_to_tensor(y_val)
    y_test_tf = tf.convert_to_tensor(y_test)

    # Convert TensorFlow tensors to NumPy arrays
    X_train_np = X_train_tf.numpy()
    X_val_np = X_val_tf.numpy()
    X_test_np = X_test_tf.numpy()
    y_train_np = y_train_tf.numpy()
    y_val_np = y_val_tf.numpy()
    y_test_np = y_test_tf.numpy()

    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train_np)
    X_val = torch.tensor(X_val_np)
    X_test = torch.tensor(X_test_np)
    y_train = torch.tensor(y_train_np)
    y_val = torch.tensor(y_val_np)
    y_test = torch.tensor(y_test_np)

    # Converting y_test to categorical data
    y_train = torch.nn.functional.one_hot(y_train, num_classes=num_classes)
    y_val = torch.nn.functional.one_hot(y_val, num_classes=num_classes)
    y_test = torch.nn.functional.one_hot(y_test, num_classes=num_classes)

    y_train = y_train.float()
    y_val = y_val.float()
    y_test = y_test.float()

    # Print shape
    print(f"X train : {X_train.shape}, {type(X_train)}")
    print(f"X val : {X_val.shape}, {type(X_val)}")
    print(f"X test : {X_test.shape}, {type(X_test)}")
    print(f"Y train : {y_train.shape}, {type(y_train)}")
    print(f"y val : {y_val.shape}, {type(y_val)}")
    print(f"y test : {y_test.shape}, {type(y_test)}")

    # Preparing Dataset
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    # Creating Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Get the shape of the first batch from train_loader
    for batch in train_loader:
        X_batch, y_batch = batch
        print(f"Shape of X_batch: {X_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")
        break  # Break after the first batch to avoid processing the entire dataset

    # Repeat for val_loader and test_loader
    for batch in val_loader:
        X_batch, y_batch = batch
        print(f"Shape of X_batch: {X_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")
        break

    for batch in test_loader:
        X_batch, y_batch = batch
        print(f"Shape of X_batch: {X_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")
        break

    return train_loader, val_loader, test_loader, num_classes, rating_counts_sorted.tolist(), vectorizer
