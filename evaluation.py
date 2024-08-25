# Loading Dataset
import argparse
import os

import numpy as np
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.lstm import LSTM
from models.rnn import RNN
from models.cnn import CNN
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score



def eval_loop(epoch, model, train_loader, loss_func, optimizer, scheduler, writer, split_type, device) :

    model.eval()

    pbar = tqdm(train_loader, desc=f"{split_type} : Epoch {epoch + 1}/{args.epochs}", unit="batch")
    running_loss = 0
    curr_count = 0
    y_true = []
    y_pred = []

    for batch_idx, (x_batch, y_batch) in enumerate(pbar) :

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward Pass
        y_hat_batch = model(x_batch)
        loss = loss_func(y_hat_batch, y_batch)

        if model.training :
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Adding to running loss, y_true and y_pred
        running_loss += loss.item()
        curr_count += x_batch.shape[0]
        y_true.append(y_batch.clone())
        y_pred.append(y_hat_batch.detach().clone())

        pbar.set_postfix({
            "Loss" : running_loss / curr_count
        })


    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)


    y_true_max = torch.argmax(y_true, dim= 1)
    y_pred_max = torch.argmax(y_pred, dim= 1)

    # Convert to numpy
    y_true = y_true.cpu().detach().clone().numpy()
    y_pred = y_pred.cpu().detach().clone().numpy()
    y_true_max = y_true_max.cpu().detach().clone().numpy()
    y_pred_max = y_pred_max.cpu().detach().clone().numpy()


    accuracy = accuracy_score(y_true_max, y_pred_max)
    balance_accuracy = balanced_accuracy_score(y_true_max, y_pred_max)
    precision = precision_score(y_true_max, y_pred_max, average='macro')
    recall = recall_score(y_true_max, y_pred_max, average='macro')
    f1 = f1_score(y_true_max, y_pred_max, average='macro')
    auc = roc_auc_score(y_true, y_pred, multi_class="ovr")

    pbar.set_postfix({
        "Loss": running_loss / curr_count,
        "Acc" : accuracy,
        "Bal Acc" : balance_accuracy,
        "AUC" : auc,
        "F1" : f1
    })

    # Adding details to tensorboard
    writer.add_scalar(f"{split_type}/loss", running_loss / curr_count, epoch + 1)
    writer.add_scalar(f"{split_type}/accuracy", accuracy, epoch + 1)
    writer.add_scalar(f"{split_type}/balance_accuracy", balance_accuracy, epoch + 1)
    writer.add_scalar(f"{split_type}/precision", precision, epoch + 1)
    writer.add_scalar(f"{split_type}/recall", recall, epoch + 1)
    writer.add_scalar(f"{split_type}/f1", f1, epoch + 1)
    writer.add_scalar(f"{split_type}/auc", auc, epoch + 1)

    return accuracy, auc, balance_accuracy, running_loss / curr_count



if __name__ == "__main__" :


    parser = argparse.ArgumentParser(description='Script for setting up training parameters.')

    # Add arguments
    parser.add_argument('--output_seq_len', type=int, default=100, help='Length of the output sequence')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--embed_size', type=int, default=300, help='Size of the embedding vector')
    parser.add_argument('--dataset_type', type=str, default='dataset_2',
                        help='We have 3 options dataset_1, dataset_2, dataset_3 '
                             'dataset_2 is Original dataset ----> shivde')
    parser.add_argument("--ckpt_dir", type=str, default="/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_evals", help="can edit any save directory")
    parser.add_argument("--bool_initialize_weights", type=bool, default=False)
    parser.add_argument("--pretrained_wv_type", type=str, default="word2vec", help="Options : word2vec, glove")
    parser.add_argument("--model_type", type=str, default="lstm", help="Options : lstm, rnn, cnn")
    # Parse arguments
    args = parser.parse_args()

    # Defining Hyper parameters
    output_seq_len = args.output_seq_len
    batch_size = args.batch_size
    embed_size = args.embed_size
    dataset_type = args.dataset_type
    learning_rate = args.learning_rate

    # Adding cuda device
    device = "cuda:3"

    # Loading model weights
    model_weights_path = "/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_ckpts/model_lstm_dataset_2_bs_32_lr_0.001_embed_300/21.pt"
    data_dict = torch.load(model_weights_path)
    vectorizer = data_dict['vectorizer']
    num_classes = data_dict['num_classes']
    rating_counts = data_dict['rating_count']
    word2idx = data_dict['word2idx']
    idx2word = data_dict['idx2word']

    # Loading Dataset
    from dataset.dataset_2_shivde import load_test_val_dataset
    test_loader = load_test_val_dataset(
        file_path=None, num_classes=num_classes, vectorizer=vectorizer, batch_size=batch_size
    )

    num_vocab = vectorizer.vocabulary_size()

    # Loading Model
    if args.model_type == "lstm" :
        # Defining Model
        model = LSTM(total_word=num_vocab,
                     embed_size=embed_size,
                     hidden_size=300,
                     num_class=num_classes,
                     vectorizer=vectorizer,
                     word2idx=word2idx,
                     idx2word=idx2word,
                     bool_initialize_weights=args.bool_initialize_weights,
                     pretrained_wv_type=args.pretrained_wv_type
                     )
    if args.model_type == "rnn" :
        # Defining Model
        model = RNN(total_word=num_vocab,
                     embed_size=embed_size,
                     hidden_size=300,
                     num_class=num_classes,
                     vectorizer=vectorizer,
                     word2idx=word2idx,
                     idx2word=idx2word,
                     bool_initialize_weights=args.bool_initialize_weights,
                     pretrained_wv_type=args.pretrained_wv_type
                     )
    if args.model_type == "cnn" :
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
    exp_name = f"model_{args.model_type}_{dataset_type}_bs_{args.batch_size}_lr_{learning_rate}_embed_{embed_size}"
    print(f"Exp name : {exp_name}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/{exp_name}')

    test_accuracy, test_auc, test_balance_accuracy, test_loss = eval_loop(model, test_loader, loss_func,
                                                                                writer, "test", device)

    final_dict = {
        'test_accuracy': test_accuracy,
        'test_bal_acc': test_balance_accuracy,
        'test_auc': test_auc,
    }

    os.makedirs(f"{args.ckpt_dir}/{exp_name}", exist_ok=True)
    torch.save(final_dict, f"{args.ckpt_dir}/{exp_name}/final.pt")
    print(f"model saved with accuracy: {test_accuracy:.2f}%")

