# Loading Dataset
import argparse
import os

import numpy as np
import torch.nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, \
    roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from models.lstm import LSTM
from models.rnn import RNN
from models.cnn import CNN
from models.cnn2 import CNN2
from models.gru import GRU


def eval_loop(model, test_loader, loss_func,  writer, split_type, device) :

    model.eval()
    epoch = 0

    pbar = tqdm(test_loader, desc=f"{split_type} ", unit="batch")
    running_loss = 0
    curr_count = 0
    y_true = []
    y_pred = []
    x_original = []

    for batch_idx, (x_batch, y_batch, x_text) in enumerate(pbar):

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward Pass
        y_hat_batch = model(x_batch)
        loss = loss_func(y_hat_batch, y_batch)

        # Adding to running loss, y_true and y_pred
        running_loss += loss.item()
        curr_count += x_batch.shape[0]
        y_true.append(y_batch.clone())
        y_pred.append(y_hat_batch.detach().clone())
        x_original += list(x_text)

        pbar.set_postfix({
            "Loss": running_loss / curr_count
        })

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    y_true_max = torch.argmax(y_true, dim=1)
    y_pred_max = torch.argmax(y_pred, dim=1)

    # Convert to numpy
    y_true = y_true.cpu().detach().clone().numpy()
    y_pred = y_pred.cpu().detach().clone().numpy()
    y_true_max = y_true_max.cpu().detach().clone().numpy()
    y_pred_max = y_pred_max.cpu().detach().clone().numpy()

    print(f"X original shape : {len(list(x_original))} and {type(list(x_original))}")
    print(f"Y prediction : {y_pred_max.shape}")

    accuracy = accuracy_score(y_true_max, y_pred_max)
    balance_accuracy = balanced_accuracy_score(y_true_max, y_pred_max)
    precision = precision_score(y_true_max, y_pred_max, average='macro')
    recall = recall_score(y_true_max, y_pred_max, average='macro')
    f1 = f1_score(y_true_max, y_pred_max, average='micro')
    auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
    auc_class = roc_auc_score(y_true, y_pred, multi_class="ovr", average=None)
    f1_class = f1_score(y_true_max, y_pred_max, average=None)
    pbar.set_postfix({
        "Loss": running_loss / curr_count,
        "Acc": accuracy,
        "Bal Acc": balance_accuracy,
        "AUC": auc,
        "F1": f1
    })

    # Adding details to tensorboard
    writer.add_scalar(f"{split_type}/loss", running_loss / curr_count, epoch)
    writer.add_scalar(f"{split_type}/accuracy", accuracy, epoch)
    writer.add_scalar(f"{split_type}/balance_accuracy", balance_accuracy, epoch)
    writer.add_scalar(f"{split_type}/precision", precision, epoch)
    writer.add_scalar(f"{split_type}/recall", recall, epoch)
    writer.add_scalar(f"{split_type}/f1", f1, epoch)
    writer.add_scalar(f"{split_type}/auc", auc, epoch)
    for idx in range(int(auc_class.shape[0])):
        writer.add_scalar(f'{split_type}_AUC/Class {idx}', auc_class[idx], epoch)
    for idx in range(int(f1_class.shape[0])):
        writer.add_scalar(f'{split_type}_F1/Class {idx}', f1_class[idx], epoch)

    return accuracy, auc, balance_accuracy, running_loss / curr_count, f1, x_original, list(y_pred_max)

def eval_loop_only_text(model, test_loader, loss_func,  writer, split_type, device) :

    model.eval()
    epoch = 0

    pbar = tqdm(test_loader, desc=f"{split_type} ", unit="batch")
    running_loss = 0
    curr_count = 0
    y_pred = []
    x_original = []

    for batch_idx, (x_batch, x_text) in enumerate(pbar):

        x_batch = x_batch.to(device)

        # Forward Pass
        y_hat_batch = model(x_batch)

        # Adding to running loss, y_true and y_pred
        curr_count += x_batch.shape[0]
        y_pred.append(y_hat_batch.detach().clone())
        x_original += list(x_text)

    y_pred = torch.cat(y_pred, dim=0)
    y_pred_max = torch.argmax(y_pred, dim=1)

    # Convert to numpy
    y_pred = y_pred.cpu().detach().clone().numpy()
    y_pred_max = y_pred_max.cpu().detach().clone().numpy()

    print(f"X original shape : {len(list(x_original))} and {type(list(x_original))}")
    print(f"Y prediction : {y_pred_max.shape}")


    return x_original, list(y_pred_max)


if __name__ == "__main__" :


    parser = argparse.ArgumentParser(description='Script for setting up training parameters.')

    # Add arguments
    parser.add_argument('--output_seq_len', type=int, default=100, help='Length of the output sequence')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--embed_size', type=int, default=300, help='Size of the embedding vector')
    parser.add_argument('--dataset_type', type=str, default='dataset_2',
                        help='We have 3 options dataset_1, dataset_2, dataset_3 '
                             'dataset_2 is Original dataset ----> shivde')
    parser.add_argument("--root_dir", type=str, default="/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/prediction_dir", help="can edit any save directory")
    parser.add_argument("--model_type", type=str, default="cnn2", help="Options : lstm, rnn, cnn")
    parser.add_argument("--only_text", type=bool, default=False, help="True : if not true label is present ")
    # Parse arguments
    args = parser.parse_args()

    # Defining Hyper parameters
    output_seq_len = args.output_seq_len
    batch_size = args.batch_size
    embed_size = args.embed_size
    dataset_type = args.dataset_type
    args.bool_initialize_weights = False
    args.pretrained_wv_type = "word2vec"
    result_file_name = "result.txt"

    # Adding cuda device
    device = "cuda:3"

    best_test_accuracy = 0
    best_test_auc = 0
    best_test_balance_accuracy = 0
    best_test_loss = 0
    best_test_f1 = 0


    # model_weights_dict = {
    #     'cnn2': ['/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_ckpts/model_cnn2_dataset_2_bs_512_lr_0.0005_embed_300_classes_5_f1_weigh/19.pt',
    #              '/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_ckpts/model_cnn2_dataset_2_bs_128_lr_0.0001_embed_300_classes_5_f1_weigh/28.pt',
    #              '/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_ckpts/model_cnn2_dataset_2_bs_512_lr_0.001_embed_300_classes_5_f1_weigh/5.pt',
    #              '/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_ckpts/model_cnn2_dataset_2_bs_128_lr_0.001_embed_300_classes_5_f1_weigh/2.pt',
    #              '/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_ckpts/model_cnn2_dataset_2_bs_128_lr_0.0005_embed_300_classes_5_f1_weigh/10.pt'],
    #     'cnn': ['/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_ckpts/model_cnn_dataset_2_bs_128_lr_0.0005_embed_300_classes_5_f1_weigh/73.pt',
    #             '/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_ckpts/model_cnn_dataset_2_bs_64_lr_0.001_embed_300_classes_5_f1_weigh/62.pt'],
    # }

    model_weights_dict = {
        "cnn2" : ['/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_ckpts/model_cnn2_dataset_2_bs_512_lr_0.0005_embed_300_classes_5_f1_weigh/19.pt']
    }

    model_types = list(model_weights_dict.keys())

    # Iterate over all the model types
    for model_type in model_types :
        for model_weights_path in model_weights_dict[model_type] :

            # Check if the current path is a file (and not a directory)
            if os.path.isfile(model_weights_path):
                print(f"Processing file: {model_weights_path}")
                # Add your processing logic here

                data_dict = torch.load(model_weights_path)
                vectorizer = data_dict['vectorizer']
                num_classes = data_dict['num_classes']
                rating_counts = data_dict['rating_count']
                word2idx = data_dict['word2idx']
                idx2word = data_dict['idx2word']

                # Loading Dataset
                from dataset.dataset_2_shivde import load_test_val_dataset, load_eval_dataset_only_text

                file_path = "/mnt/hdd/karmpatel/naman/demo/DLNLP_Ass1_Data/Aug24-Assignment1-Validation-Dataset1.csv"

                if args.only_text :
                    test_loader = load_eval_dataset_only_text(
                    file_path=file_path, num_classes=num_classes, vectorizer=vectorizer, batch_size=batch_size, original_col_name= "Original", include_original_text= True
                    )
                else :
                    test_loader = load_test_val_dataset(
                    file_path=file_path, num_classes=num_classes, vectorizer=vectorizer, batch_size=batch_size, original_col_name= "Original", include_original_text= True
                    )

                num_vocab = vectorizer.vocabulary_size()

                # Loading Model
                if args.model_type == "lstm":
                    # Defining Model
                    model = LSTM(total_word=num_vocab,
                                 embed_size=embed_size,
                                 hidden_size=300,
                                 num_class=num_classes,
                                 vectorizer=vectorizer,
                                 word2idx=word2idx,
                                 idx2word=idx2word,
                                 bool_initialize_weights=args.bool_initialize_weights,
                                 pretrained_wv_type=args.pretrained_wv_type,
                                 num_layers=1
                                 )
                if args.model_type == "rnn":
                    # Defining Model
                    model = RNN(total_word=num_vocab,
                                embed_size=embed_size,
                                hidden_size=300,
                                num_class=num_classes,
                                vectorizer=vectorizer,
                                word2idx=word2idx,
                                idx2word=idx2word,
                                bool_initialize_weights=args.bool_initialize_weights,
                                pretrained_wv_type=args.pretrained_wv_type,
                                num_layers=1
                                )
                if args.model_type == "cnn":
                    # Defining Model
                    model = CNN(total_word=num_vocab,
                                embed_size=embed_size,
                                hidden_size=300,
                                num_class=num_classes,
                                vectorizer=vectorizer,
                                word2idx=word2idx,
                                idx2word=idx2word,
                                bool_initialize_weights=args.bool_initialize_weights,
                                pretrained_wv_type=args.pretrained_wv_type
                                )
                if args.model_type == "cnn2":
                    # Defining Model
                    model = CNN2(total_word=num_vocab,
                                 embed_size=embed_size,
                                 hidden_size=300,
                                 num_class=num_classes,
                                 vectorizer=vectorizer,
                                 word2idx=word2idx,
                                 idx2word=idx2word,
                                 bool_initialize_weights=args.bool_initialize_weights,
                                 pretrained_wv_type=args.pretrained_wv_type
                                 )
                if args.model_type == "lstm2":
                    # Defining Model
                    model = LSTM(total_word=num_vocab,
                                 embed_size=embed_size,
                                 hidden_size=300,
                                 num_class=num_classes,
                                 vectorizer=vectorizer,
                                 word2idx=word2idx,
                                 idx2word=idx2word,
                                 bool_initialize_weights=args.bool_initialize_weights,
                                 pretrained_wv_type=args.pretrained_wv_type,
                                 num_layers=2
                                 )
                if args.model_type == "rnn2":
                    # Defining Model
                    model = RNN(total_word=num_vocab,
                                embed_size=embed_size,
                                hidden_size=300,
                                num_class=num_classes,
                                vectorizer=vectorizer,
                                word2idx=word2idx,
                                idx2word=idx2word,
                                bool_initialize_weights=args.bool_initialize_weights,
                                pretrained_wv_type=args.pretrained_wv_type,
                                num_layers=2
                                )
                if args.model_type == "lstm3":
                    # Defining Model
                    model = LSTM(total_word=num_vocab,
                                 embed_size=embed_size,
                                 hidden_size=300,
                                 num_class=num_classes,
                                 vectorizer=vectorizer,
                                 word2idx=word2idx,
                                 idx2word=idx2word,
                                 bool_initialize_weights=args.bool_initialize_weights,
                                 pretrained_wv_type=args.pretrained_wv_type,
                                 num_layers=3
                                 )
                if args.model_type == "rnn3":
                    # Defining Model
                    model = RNN(total_word=num_vocab,
                                embed_size=embed_size,
                                hidden_size=300,
                                num_class=num_classes,
                                vectorizer=vectorizer,
                                word2idx=word2idx,
                                idx2word=idx2word,
                                bool_initialize_weights=args.bool_initialize_weights,
                                pretrained_wv_type=args.pretrained_wv_type,
                                num_layers=3
                                )
                if args.model_type == "gru":
                    # Defining Model
                    model = GRU(total_word=num_vocab,
                                embed_size=embed_size,
                                hidden_size=300,
                                num_class=num_classes,
                                vectorizer=vectorizer,
                                word2idx=word2idx,
                                idx2word=idx2word,
                                bool_initialize_weights=args.bool_initialize_weights,
                                pretrained_wv_type=args.pretrained_wv_type,
                                num_layers=1
                                )
                if args.model_type == "gru2":
                    # Defining Model
                    model = GRU(total_word=num_vocab,
                                embed_size=embed_size,
                                hidden_size=300,
                                num_class=num_classes,
                                vectorizer=vectorizer,
                                word2idx=word2idx,
                                idx2word=idx2word,
                                bool_initialize_weights=args.bool_initialize_weights,
                                pretrained_wv_type=args.pretrained_wv_type,
                                num_layers=2
                                )
                if args.model_type == "gru3":
                    # Defining Model
                    model = GRU(total_word=num_vocab,
                                embed_size=embed_size,
                                hidden_size=300,
                                num_class=num_classes,
                                vectorizer=vectorizer,
                                word2idx=word2idx,
                                idx2word=idx2word,
                                bool_initialize_weights=args.bool_initialize_weights,
                                pretrained_wv_type=args.pretrained_wv_type,
                                num_layers=3
                                )

                # Load model weights
                model_weights = data_dict['model_weights']
                model.load_state_dict(model_weights)
                model.to(device)

                # Generating Class weights for model
                counts = np.array(rating_counts)
                class_weights = sum(rating_counts) / (len(rating_counts) * counts)
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
                class_weights = class_weights.to(device)
                print(f"Class Weights: {class_weights}")

                # Defining Loss, Optimizer, Scheduler
                # loss_func = torch.nn.CrossEntropyLoss(weight= class_weights)
                loss_func = torch.nn.CrossEntropyLoss()

                # Creating exp name
                exp_name = f"model_{model_type}_{dataset_type}_bs_{args.batch_size}_embed_{embed_size}"
                if args.only_text :
                    exp_name += "_only_text"
                print(f"Exp name : {exp_name}")

                # Initialize TensorBoard writer
                writer = SummaryWriter(f'eval/{exp_name}')

                if args.only_text:
                    x_original, y_prediction = eval_loop_only_text(model, test_loader,
                        loss_func,
                        writer, "test", device)

                    # Saving details in file
                    df = pd.DataFrame(y_prediction, columns=['Prediction'])
                    df = pd.DataFrame(
                        {
                            'Original Text': x_original,
                            'Prediction': y_prediction
                        }
                    )
                    df['Prediction'] += 1
                    # Save the DataFrame to an Excel file
                    df.to_excel(f'{args.root_dir}/{exp_name}_prediction_only_text.xlsx', index=False)

                    print(f"Saving Prediction : {exp_name} ")


                else :

                    test_accuracy, test_auc, test_balance_accuracy, test_loss, test_f1, x_original, y_prediction = eval_loop(
                        model, test_loader,
                        loss_func,
                        writer, "test", device)

                    if best_test_f1 < test_f1:
                        best_test_f1 = test_f1

                        results_dict = {
                            "Accuracy": test_accuracy,
                            "AUC": test_auc,
                            "Balance Accuracy": test_balance_accuracy,
                            "Test Loss": test_loss,
                            "Test F1": test_f1
                        }

                        # Saving results into Result file name
                        # Open the file in append mode
                        with open(result_file_name, 'a') as f:
                            f.write(f"-----------Exp Name : {exp_name}  \n")
                            # Loop through the dictionary keys
                            for key in results_dict.keys():
                                # Write the key to the file followed by a newline
                                f.write(f"{key}\t{results_dict[key]}\n")
                            f.write(f"-----------" * 10)
                            f.write("\n")

                        # Saving details in file
                        df = pd.DataFrame(y_prediction, columns=['Prediction'])
                        df = pd.DataFrame(
                            {
                                'Original Text': x_original,
                                'Prediction': y_prediction
                            }
                        )
                        df['Prediction'] += 1
                        # Save the DataFrame to an Excel file
                        df.to_excel(f'{args.root_dir}/{exp_name}_prediction.xlsx', index=False)

                        print(f"Saving Prediction : {exp_name} ")


