import os
import string
import typing

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class CustomDatasetWithOriginal(Dataset):
    def __init__(self, features, labels, original_text):
        self.features = features
        self.labels = labels
        self.original_test = original_text

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.original_test[idx]

class CustomDatasetOnlyText(Dataset):
    def __init__(self, features, original_text):
        self.features = features
        self.original_test = original_text

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.original_test[idx]


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

def rating2c(score) :

    if int(score) == 5 :
        return  1
    else :
        return 0

def rating3c(score) :

    # Logic is score of 0/1 ---> Negative, 3 ---> Neutral, 4/5 ----> Positive
    if int(score) < 3 :
        return 0
    elif int(score) == 3 :
        return 1
    else :
        return 2

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

    sentence = ' '.join(sentence)
    return str(sentence)


def text_preprocessing(df : pd.DataFrame, get2class = False) :

    # Preprocessing rating
    df['Original'] = df['Review']
    df['Rating'] = df['Rating'].apply(rating)
    if get2class :
        df['Rating2C'] = df['Rating'].apply(rating2c)
        df['Rating3C'] = df['Rating'].apply(rating3c)

    df['Rating'] = df['Rating'] - df['Rating'].min()
    # Cleaning Review Text
    df['Review'] = df['Review'].apply(clean_text)

    # Length of Dataset
    df['Length'] = df['Review'].apply(lambda r: len(r.split(" ")))
    new_length = df['Length'].sum()
    print('Total word after cleaning: {}'.format(new_length))
    return df

def text_preprocessing_only_text(df : pd.DataFrame) :

    # Preprocessing rating
    df['Original'] = df['Review']
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

def preprocessing(df : pd.DataFrame, dir_path = "../data", file_name = None, get2class = False, visulaize = True) :

    if file_name is None :
        print("Enter a valid file name")
        exit()

    if visulaize :
        data_visulaization(df)
        # Preprocessing dataset
        df = text_preprocessing(df, get2class = True)
    else :
        # Preprocessing dataset
        df = text_preprocessing_only_text(df)
    processed_file_path = dataframe_to_csv(df, dir_path, file_name)
    return processed_file_path

def split_train_test_validation(df : pd.DataFrame, x_col = "Review", y_col = "Rating", valid_test_ratio =0.2 , valid_ratio_from_test = 0.5) :

    df[x_col] = df[x_col].astype(str)
    df[y_col] = df[y_col].astype(int)

    # Step 1: Split into training and temporary sets (temporary = test + validation)
    X_train, X_temp, y_train, y_temp = train_test_split(
        df[x_col], df[y_col], test_size= valid_test_ratio, stratify=df[y_col], random_state=42,
    )

    # Step 2: Split the temporary set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size= valid_ratio_from_test, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data_loaders(file_path : str, valid_test_ration =0.2, valid_ratio_from_test =0.5,
                     output_seq_len = None, batch_size = 64, x_col = "Review", y_col = "Rating" , split_sub_class = False) :

    '''

    :param file_path: Datatype : str Path contains all the data which is split into 3 parts train, val, test
    :param valid_test_ration:
    :param valid_ratio_from_test:
    :param output_seq_len:
    :param batch_size:
    :return:
    '''

    # Using Post processed csv file
    print(f"Reading data from : {file_path} ")
    df = pd.read_csv(file_path)

    if split_sub_class :
        df = df[df['Rating2C'] == 0]

    # Get details
    unique_ratings = df[y_col].unique()
    num_classes = len(unique_ratings)
    print(f"Number of classes : {num_classes}")
    rating_counts = df[y_col].value_counts()
    rating_counts_sorted = rating_counts.sort_index()

    # Creating max sequence length ( Mean + 2 * std)
    if output_seq_len is None :
        output_seq_len = int(df['Length'].mean() + 2 * df['Length'].std())

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_validation(df, x_col = x_col, y_col = y_col)

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

    # Now creating weighted sampler (because needed y_train, y_val, y_test in label representation form)
    class_weights = 1. / torch.tensor(rating_counts_sorted, dtype=torch.float)
    class_weights /= class_weights.sum()
    train_sample_weights = class_weights[y_train]
    val_sample_weights = class_weights[y_val]
    test_sample_weights = class_weights[y_test]
    print(class_weights)


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


    # Create a WeightedRandomSampler
    train_sampler = WeightedRandomSampler(weights= train_sample_weights, num_samples=len(train_sample_weights), replacement=True)
    val_sampler = WeightedRandomSampler(weights= val_sample_weights, num_samples= len(val_sample_weights), replacement=True)
    test_sampler = WeightedRandomSampler(weights= test_sample_weights, num_samples= len(test_sample_weights), replacement=True)

    # Preparing Dataset
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    # Creating Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler= train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Get the shape of the first batch from train_loader
    for batch in train_loader:
        X_batch, y_batch = batch
        print(f"Shape of X_batch: {X_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")
        unique, counts = torch.unique(torch.argmax(y_batch, dim=1), return_counts=True)
        print(f"Training Batch class distribution: {dict(zip(unique.numpy(), counts.numpy()))}")
        break  # Break after the first batch to avoid processing the entire dataset

    # Repeat for val_loader and test_loader
    for batch in val_loader:
        X_batch, y_batch = batch
        print(f"Shape of X_batch: {X_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")
        unique, counts = torch.unique(torch.argmax(y_batch, dim=1), return_counts=True)
        print(f"Validation Batch class distribution: {dict(zip(unique.numpy(), counts.numpy()))}")
        break

    for batch in test_loader:
        X_batch, y_batch = batch
        print(f"Shape of X_batch: {X_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")
        unique, counts = torch.unique(torch.argmax(y_batch, dim=1), return_counts=True)
        print(f"Test Batch class distribution: {dict(zip(unique.numpy(), counts.numpy()))}")
        break

    return train_loader, val_loader, test_loader, num_classes, rating_counts_sorted.tolist(), vectorizer

def get_data_loaders_with_different_file_paths(file_path : typing.Dict, output_seq_len=None, batch_size=64):
    '''

    :param file_path: Datatype dict : Key can be train, val, test and Values : file path
    :param output_seq_len:
    :param batch_size:
    :return:
    '''

    # Using Post processed csv file
    print(f"Reading data from : {file_path['train']} ")
    df_train = pd.read_csv(file_path['train'])

    print(f"Reading data from : {file_path['val']} ")
    df_val = pd.read_csv(file_path['val'])

    print(f"Reading data from : {file_path['test']} ")
    df_test = pd.read_csv(file_path['test'])

    # Get details
    unique_ratings = df_train['Rating'].unique()
    num_classes = len(unique_ratings)
    print(f"Number of classes : {num_classes}")
    rating_counts = df_train['Rating'].value_counts()
    rating_counts_sorted = rating_counts.sort_index()

    # Creating max sequence length ( Mean + 2 * std)
    if output_seq_len is None :
        output_seq_len = int(df_train['Length'].mean() + 2 * df_train['Length'].std())

    # Now get the train, validation, test
    X_train, y_train = df_train['Review'], df_train['Rating']
    X_val, y_val = df_val['Review'], df_val['Rating']
    X_test, y_test = df_test['Review'], df_test['Rating']

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

    # Now creating weighted sampler (because needed y_train, y_val, y_test in label representation form)
    class_weights = 1. / torch.tensor(rating_counts_sorted, dtype=torch.float)
    class_weights /= class_weights.sum()
    train_sample_weights = class_weights[y_train]
    val_sample_weights = class_weights[y_val]
    test_sample_weights = class_weights[y_test]
    print(class_weights)

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

    # Create a WeightedRandomSampler
    train_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights),
                                          replacement=True)
    val_sampler = WeightedRandomSampler(weights=val_sample_weights, num_samples=len(val_sample_weights),
                                        replacement=True)
    test_sampler = WeightedRandomSampler(weights=test_sample_weights, num_samples=len(test_sample_weights),
                                         replacement=True)

    # Preparing Dataset
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    # Creating Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler= train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Get the shape of the first batch from train_loader
    for batch in train_loader:
        X_batch, y_batch = batch
        print(f"Shape of X_batch: {X_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")
        unique, counts = torch.unique(torch.argmax(y_batch, dim=1), return_counts=True)
        print(f"Training Batch class distribution: {dict(zip(unique.numpy(), counts.numpy()))}")
        break  # Break after the first batch to avoid processing the entire dataset

    # Repeat for val_loader and test_loader
    for batch in val_loader:
        X_batch, y_batch = batch
        print(f"Shape of X_batch: {X_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")
        unique, counts = torch.unique(torch.argmax(y_batch, dim=1), return_counts=True)
        print(f"Validation Batch class distribution: {dict(zip(unique.numpy(), counts.numpy()))}")
        break

    for batch in test_loader:
        X_batch, y_batch = batch
        print(f"Shape of X_batch: {X_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")
        unique, counts = torch.unique(torch.argmax(y_batch, dim=1), return_counts=True)
        print(f"Test Batch class distribution: {dict(zip(unique.numpy(), counts.numpy()))}")
        break

    return train_loader, val_loader, test_loader, num_classes, rating_counts_sorted.tolist(), vectorizer

def get_train_data_loaders(file_path, batch_size = 64, output_seq_len = None,
                           x_col = "Review", y_col = "Rating", split_sub_class = False) :

    # Using Post processed csv file
    print(f"Reading data from : {file_path} ")
    df_train = pd.read_csv(file_path)

    if split_sub_class :
        df_train = df_train[df_train['Rating2C'] == 0]

    # Get details
    unique_ratings = df_train[y_col].unique()
    num_classes = len(unique_ratings)
    print(f"Number of classes : {num_classes}")
    rating_counts = df_train[y_col].value_counts()
    rating_counts_sorted = rating_counts.sort_index()


    # Creating max sequence length ( Mean + 2 * std)
    if output_seq_len is None:
        output_seq_len = int(df_train['Length'].mean() + 3 * df_train['Length'].std())

    # Now get the train, validation, test
    X_train, y_train = df_train[x_col].astype(str), df_train[y_col]

    # Check the size sets
    print(f"Training set: {len(X_train)} samples")

    # Building Vocabulary, convert text -> number
    vectorizer = TextVectorization(
        max_tokens=None,
        output_mode="int",
        output_sequence_length=output_seq_len
    )

    vectorizer.adapt(X_train)

    X_train = vectorizer(X_train)
    # Convert them to tensor
    X_train_tf = tf.convert_to_tensor(X_train)
    y_train_tf = tf.convert_to_tensor(y_train)

    # Convert TensorFlow tensors to NumPy arrays
    X_train_np = X_train_tf.numpy()
    y_train_np = y_train_tf.numpy()

    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train_np)
    y_train = torch.tensor(y_train_np)

    # Now creating weighted sampler (because needed y_train, y_val, y_test in label representation form)
    class_weights = 1. / torch.tensor(rating_counts_sorted, dtype=torch.float)
    class_weights /= class_weights.sum()
    train_sample_weights = class_weights[y_train]
    print(class_weights)

    # Converting y_test to categorical data
    y_train = torch.nn.functional.one_hot(y_train, num_classes=num_classes)
    y_train = y_train.float()

    # Print shape
    print(f"X train : {X_train.shape}, {type(X_train)}")
    print(f"Y train : {y_train.shape}, {type(y_train)}")

    # Create a WeightedRandomSampler
    train_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights),
                                          replacement=True)

    # Preparing Dataset
    train_dataset = CustomDataset(X_train, y_train)

    # Creating Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler= train_sampler)

    # Get the shape of the first batch from train_loader
    for batch in train_loader:
        X_batch, y_batch = batch
        print(f"Shape of X_batch: {X_batch.shape}")
        print(f"Shape of y_batch: {y_batch.shape}")
        unique, counts = torch.unique(torch.argmax(y_batch, dim=1), return_counts=True)
        print(f"Training Batch class distribution: {dict(zip(unique.numpy(), counts.numpy()))}")
        break  # Break after the first batch to avoid processing the entire dataset

    return train_loader, num_classes, rating_counts_sorted.tolist(), vectorizer

def get_eval_data_loaders(file_path, num_classes, vectorizer, batch_size = 64,
                          x_col = "Review", y_col = "Rating" , split_sub_class = False, original_col_name = "Original", include_original_text = False) :

    if include_original_text :

        # Using Post processed csv file
        print(f"Reading data from : {file_path} ")
        df = pd.read_csv(file_path)

        if split_sub_class:
            df = df[df['Rating2C'] == 0]

        X, X_original, y = df[x_col].astype(str), df[original_col_name], df[y_col]

        # Check the size sets
        print(f"Dataset: {len(X)} samples")

        # convert text -> number
        X = vectorizer(X)
        X_tf = tf.convert_to_tensor(X)
        y_tf = tf.convert_to_tensor(y)

        # Convert Tensorflow tensors to NumPy arrays
        X_np = X_tf.numpy()
        y_np = y_tf.numpy()

        # Convert NumPy arrays to PyTorch tensors
        X = torch.tensor(X_np)
        y = torch.tensor(y_np)

        # Converting y_test to categorical data
        y = torch.nn.functional.one_hot(y, num_classes=num_classes)
        y = y.float()

        # Print shape
        print(f"X : {X.shape}, {type(X)}")
        print(f"y : {y.shape}, {type(y)}")

        # X_original ---> panda seris to list
        X_original = X_original.tolist()

        # Preparing Dataset
        dataset = CustomDatasetWithOriginal(X, y, X_original)

        # Creating Dataloaders
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Get the shape of the first batch from train_loader
        for batch in data_loader:
            X_batch, y_batch, X_original = batch
            print(f"Shape of X_batch: {X_batch.shape}")
            print(f"Shape of y_batch: {y_batch.shape}")
            print(f"Shape of X_original: {X_original}")
            unique, counts = torch.unique(torch.argmax(y_batch, dim=1), return_counts=True)
            print(f"Batch class distribution: {dict(zip(unique.numpy(), counts.numpy()))}")
            break  # Break after the first batch to avoid processing the entire dataset

    else :
        # Using Post processed csv file
        print(f"Reading data from : {file_path} ")
        df = pd.read_csv(file_path)

        if split_sub_class:
            df = df[df['Rating2C'] == 0]

        X, y = df[x_col].astype(str), df[y_col]

        # Check the size sets
        print(f"Dataset: {len(X)} samples")

        # convert text -> number
        X = vectorizer(X)
        X_tf = tf.convert_to_tensor(X)
        y_tf = tf.convert_to_tensor(y)

        # Convert Tensorflow tensors to NumPy arrays
        X_np = X_tf.numpy()
        y_np = y_tf.numpy()

        # Convert NumPy arrays to PyTorch tensors
        X = torch.tensor(X_np)
        y = torch.tensor(y_np)

        # Converting y_test to categorical data
        y = torch.nn.functional.one_hot(y, num_classes=num_classes)
        y = y.float()

        # Print shape
        print(f"X : {X.shape}, {type(X)}")
        print(f"y : {y.shape}, {type(y)}")

        # Preparing Dataset
        dataset = CustomDataset(X, y)

        # Creating Dataloaders
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Get the shape of the first batch from train_loader
        for batch in data_loader:
            X_batch, y_batch = batch
            print(f"Shape of X_batch: {X_batch.shape}")
            print(f"Shape of y_batch: {y_batch.shape}")
            unique, counts = torch.unique(torch.argmax(y_batch, dim=1), return_counts=True)
            print(f"Batch class distribution: {dict(zip(unique.numpy(), counts.numpy()))}")
            break  # Break after the first batch to avoid processing the entire dataset


    return data_loader


def get_eval_data_loaders_only_text(file_path, num_classes, vectorizer, batch_size = 64,
                          x_col = "Review", split_sub_class = False, original_col_name = "Original", include_original_text = False) :

    if include_original_text :

        # Using Post processed csv file
        print(f"Reading data from : {file_path} ")
        df = pd.read_csv(file_path)

        if split_sub_class:
            df = df[df['Rating2C'] == 0]

        X, X_original = df[x_col].astype(str), df[original_col_name]

        # Check the size sets
        print(f"Dataset: {len(X)} samples")

        # convert text -> number
        X = vectorizer(X)
        X_tf = tf.convert_to_tensor(X)

        # Convert Tensorflow tensors to NumPy arrays
        X_np = X_tf.numpy()

        # Convert NumPy arrays to PyTorch tensors
        X = torch.tensor(X_np)


        # Print shape
        print(f"X : {X.shape}, {type(X)}")

        # X_original ---> panda seris to list
        X_original = X_original.tolist()

        # Preparing Dataset
        dataset = CustomDatasetOnlyText(X, X_original)

        # Creating Dataloaders
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Get the shape of the first batch from train_loader
        for batch in data_loader:
            X_batch, X_original = batch
            print(f"Shape of X_batch: {X_batch.shape}")
            print(f"Shape of X_original: {X_original}")
            break  # Break after the first batch to avoid processing the entire dataset

    else :
        # Using Post processed csv file
        print(f"Reading data from : {file_path} ")
        df = pd.read_csv(file_path)

        if split_sub_class:
            df = df[df['Rating2C'] == 0]

        X, y = df[x_col].astype(str), df[y_col]

        # Check the size sets
        print(f"Dataset: {len(X)} samples")

        # convert text -> number
        X = vectorizer(X)
        X_tf = tf.convert_to_tensor(X)
        y_tf = tf.convert_to_tensor(y)

        # Convert Tensorflow tensors to NumPy arrays
        X_np = X_tf.numpy()
        y_np = y_tf.numpy()

        # Convert NumPy arrays to PyTorch tensors
        X = torch.tensor(X_np)
        y = torch.tensor(y_np)

        # Converting y_test to categorical data
        y = torch.nn.functional.one_hot(y, num_classes=num_classes)
        y = y.float()

        # Print shape
        print(f"X : {X.shape}, {type(X)}")
        print(f"y : {y.shape}, {type(y)}")

        # Preparing Dataset
        dataset = CustomDataset(X, y)

        # Creating Dataloaders
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Get the shape of the first batch from train_loader
        for batch in data_loader:
            X_batch, y_batch = batch
            print(f"Shape of X_batch: {X_batch.shape}")
            print(f"Shape of y_batch: {y_batch.shape}")
            unique, counts = torch.unique(torch.argmax(y_batch, dim=1), return_counts=True)
            print(f"Batch class distribution: {dict(zip(unique.numpy(), counts.numpy()))}")
            break  # Break after the first batch to avoid processing the entire dataset


    return data_loader


def get_mapping(vectorizer) :

    '''

    :param vectorizer: Object of tf.keras.layers.Vectorization will contain all the vectors
    :return: Will return 2 dict,  word2idx and idx2word
    '''
    # Now create Mapping word --> idx
    # Reverse Mapping    idx ---> word
    word2idx = {}
    idx2word = {}

    vocab_list = vectorizer.get_vocabulary()
    for word in vocab_list:
        if word == "":
            continue
        idx = vectorizer(word)
        idx = int(idx[0])
        word2idx[word] = idx
        idx2word[idx] = word

    return word2idx, idx2word