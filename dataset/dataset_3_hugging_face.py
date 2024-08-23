import pandas as pd

from dataset.dataset_utils import preprocessing, get_data_loaders


def read_dataset_n_modify_column_name(data_dir) :

    splits = {'train': 'train.parquet',
              'validation': 'validation.parquet',
              'test': 'test.parquet'}
    df_train = pd.read_parquet(f"{data_dir}/" + splits["train"])
    df_val = pd.read_parquet(f"{data_dir}/" + splits["validation"])
    print(f"Train DataFrame Shape: {df_train.shape}")
    print(f"Validation DataFrame Shape: {df_val.shape}")

    df = pd.concat([df_train, df_val], ignore_index=True)
    print(f"DataFrame Shape: {df.shape}")

    # Column names
    column_names = df.columns
    print(f"Column names : {column_names}")
    # If column names are different let them convert to
    # "Rating" and "Review"
    print("Dataframe information")
    df = df.rename(columns={'label': 'Rating', 'sentence': 'Review'})
    print(df.info())
    return df

def load_dataset() :
    # data_dir = "/mnt/hdd/karmpatel/naman/demo/DLNLP_Ass1_Data/sst2"
    # df = read_dataset_n_modify_column_name(data_dir)
    #
    # # Preprocessing
    # file_path = preprocessing(
    #     df=df,
    #     dir_path="../data",
    #     file_name="dataset_sst2_postprocess.csv"
    # )

    file_path = f"../data/dataset_sst2_postprocess.csv"

    train_loader, val_loader, test_loader, num_classes, rating_counts, vectorizer = get_data_loaders(file_path,
                                                                                      output_seq_len=100,
                                                                                      batch_size=64)
    return train_loader, val_loader, test_loader, num_classes, rating_counts, vectorizer

if __name__ == "__main__" :

    load_dataset()


