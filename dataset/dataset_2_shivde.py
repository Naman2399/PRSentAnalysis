import pandas as pd

from dataset.dataset_utils import get_data_loaders, get_mapping, preprocessing, get_eval_data_loaders, \
    get_train_data_loaders


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
    # file_name_train = "/mnt/hdd/karmpatel/naman/demo/DLNLP_Ass1_Data/Aug24-Assignmen1-Dataset1.csv"
    # file_name_val = "/mnt/hdd/karmpatel/naman/demo/DLNLP_Ass1_Data/Aug24-Assignment1-Validation-Dataset1.csv"
    # file_name_test = "/mnt/hdd/karmpatel/naman/demo/DLNLP_Ass1_Data/Aug24-Assignment1-Validation-Dataset1.csv"
    # df_train = read_dataset_n_modify_column_name(file_name_train)
    # df_val = read_dataset_n_modify_column_name(file_name_val)
    # df_test = read_dataset_n_modify_column_name(file_name_test)
    #
    # # Preprocessing
    # file_path_train = preprocessing(
    #     df=df_train,
    #     dir_path="../data",
    #     file_name="dataset_shivde_postprocess_train.csv",
    #     get2class = True
    # )
    #
    # file_path_val = preprocessing(
    #     df=df_val,
    #     dir_path="../data",
    #     file_name="dataset_shivde_postprocess_val.csv",
    #     get2class=True
    # )
    #
    # file_path_test = preprocessing(
    #     df=df_test,
    #     dir_path="../data",
    #     file_name="dataset_shivde_postprocess_test.csv",
    #     get2class=True
    # )

    file_path_train = f"/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/data/dataset_shivde_postprocess_train.csv"
    file_path_val = f"/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/data/dataset_shivde_postprocess_val.csv"
    file_path_test = f"/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/data/dataset_shivde_postprocess_test.csv"

    x_col = "Review"
    y_col = "Rating"

    train_loader, num_classes, rating_counts, vectorizer = get_train_data_loaders(file_path_train, batch_size=batch_size, x_col = x_col, y_col = y_col)
    val_loader = get_eval_data_loaders(file_path_val, num_classes, vectorizer, batch_size=batch_size, x_col = x_col, y_col = y_col)
    test_loader = get_eval_data_loaders(file_path_test, num_classes, vectorizer, batch_size=batch_size, x_col = x_col, y_col = y_col)


    print("-" * 50)

    # train_loader, num_classes, rating_counts, vectorizer = get_train_data_loaders(file_path_train,batch_size=batch_size, x_col="Review",y_col="Rating", split_sub_class= False)
    # val_loader = get_eval_data_loaders(file_path_val, num_classes, vectorizer, batch_size=batch_size, x_col="Review", y_col="Rating", split_sub_class= False)
    # test_loader = get_eval_data_loaders(file_path_test, num_classes, vectorizer, batch_size=batch_size, x_col="Review", y_col="Rating", split_sub_class= False)


    # word2idx, idx2word = get_mapping(vectorizer)
    word2idx = None
    idx2word = None

    return train_loader, val_loader, test_loader, num_classes, rating_counts, vectorizer, word2idx, idx2word


def load_test_val_dataset(file_path, num_classes, vectorizer, batch_size) :

    df = read_dataset_n_modify_column_name(file_path)

    # Preprocessing
    file_path = preprocessing(
        df=df,
        dir_path="data",
        file_name="dataset_shivde_postprocess_eval.csv"
    )

    data_loader = get_eval_data_loaders(file_path, num_classes, vectorizer, batch_size=batch_size)

    return data_loader

if __name__ == "__main__":
    load_dataset(80, 32)








