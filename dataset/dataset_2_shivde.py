import pandas as pd

from dataset.dataset_utils import get_data_loaders, get_mapping, preprocessing


def read_dataset_n_modify_column_name(file_name) :

    df = pd.read_csv(file_name)
    # Column names
    column_names = df.columns
    print(f"Column names : {column_names}")
    # If column names are different let them convert to
    # "Rating" and "Review"
    print("Dataframe information")
    df = df.rename(columns={'overall': 'Rating', 'reviewText': 'Review'})
    print(df.info())
    return df


def load_dataset(output_seq_len, batch_size):
    file_name = "/mnt/hdd/karmpatel/naman/demo/DLNLP_Ass1_Data/Aug24-Assignmen1-Dataset1.csv"
    df = read_dataset_n_modify_column_name(file_name)

    # Preprocessing
    file_path = preprocessing(
        df=df,
        dir_path="data",
        file_name="dataset_shivde_postprocess.csv"
    )

    # file_path = f"/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/data/dataset_shivde_postprocess.csv"

    train_loader, val_loader, test_loader, num_classes, rating_counts, vectorizer = get_data_loaders(file_path,
                                                                                      batch_size= batch_size)

    word2idx, idx2word = get_mapping(vectorizer)

    return train_loader, val_loader, test_loader, num_classes, rating_counts, vectorizer, word2idx, idx2word


if __name__ == "__main__":
    load_dataset()








