import pandas as pd

from dataset.dataset_utils import preprocessing, get_train_data_loaders, get_eval_data_loaders, \
    get_mapping


def read_dataset_n_modify_column_name(data_dir, test_data) :

    splits = {'train': 'train.parquet',
              'validation': 'validation.parquet'}
    df_train = pd.read_parquet(f"{data_dir}/" + splits["train"])
    df_val = pd.read_parquet(f"{data_dir}/" + splits["validation"])
    df_test = pd.read_csv(test_data)

    print(f"Train DataFrame Shape: {df_train.shape}")
    print(f"Validation DataFrame Shape: {df_val.shape}")
    print(f"Test DataFrame Shape : {df_test.shape}")

    # Column names
    column_names = df_train.columns
    print(f"Column names : {column_names}")
    # If column names are different let them convert to
    # "Rating" and "Review"
    print("Dataframe information")
    df_train = df_train.rename(columns={'label': 'Rating', 'sentence': 'Review'})
    df_val = df_val.rename(columns={'label': 'Rating', 'sentence': 'Review'})
    df_test = df_test.rename(columns={'label': 'Rating', 'sentence': 'Review'})

    return df_train, df_val, df_test

def load_dataset(output_seq_len, batch_size) :

    # data_dir = "/mnt/hdd/karmpatel/naman/demo/DLNLP_Ass1_Data/sst2"
    # test_data = "/mnt/hdd/karmpatel/naman/demo/DLNLP_Ass1_Data/SST2_TestData.csv"
    # df_train, df_val, df_test = read_dataset_n_modify_column_name(data_dir, test_data)
    #
    # # Preprocessing
    # file_path_train = preprocessing(
    #     df=df_train,
    #     dir_path="../data",
    #     file_name="dataset_sst2_postprocess_train.csv"
    # )
    #
    # file_path_val = preprocessing(
    #     df=df_val,
    #     dir_path="../data",
    #     file_name="dataset_sst2_postprocess_val.csv"
    # )
    #
    # file_path_test = preprocessing(
    #     df=df_test,
    #     dir_path="../data",
    #     file_name="dataset_sst2_postprocess_test.csv"
    # )
    #
    # file_path = {
    #     "train" : file_path_train,
    #     "val" : file_path_val,
    #     "test" : file_path_test
    # }

    file_path_train = "/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/data/dataset_sst2_postprocess_train.csv"
    file_path_val = "/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/data/dataset_sst2_postprocess_val.csv"
    file_path_test = "/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/data/dataset_sst2_postprocess_test.csv"

    file_path = {
        "train": file_path_train,
        "val": file_path_val,
        "test": file_path_test
    }

    x_col = "Review"
    y_col = "Rating"
    train_loader, num_classes, rating_counts, vectorizer = get_train_data_loaders(file_path['train'], batch_size=batch_size, x_col= x_col, y_col= y_col)
    val_loader = get_eval_data_loaders(file_path['val'], num_classes, vectorizer, batch_size= batch_size, x_col= x_col, y_col= y_col)
    test_loader = get_eval_data_loaders(file_path['test'], num_classes, vectorizer, batch_size= batch_size, x_col= x_col, y_col= y_col)

    # word2idx, idx2word = get_mapping(vectorizer)
    word2idx = None
    idx2word = None

    return train_loader, val_loader, test_loader, num_classes, rating_counts, vectorizer, word2idx, idx2word

if __name__ == "__main__" :

    load_dataset(100, 32)


