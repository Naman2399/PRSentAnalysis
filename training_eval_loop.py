# Loading Dataset
import argparse
import os

import numpy as np
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from gpu_util import check_gpu_availability
from models.lstm import LSTM
from models.rnn import RNN
from models.cnn import CNN
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score



def train_eval_loop(epoch, model, train_loader, loss_func, optimizer, scheduler, writer, split_type, device) :

    if split_type == "train" :
        model.train()
    elif split_type == "val" or split_type == "test" :
        model.eval()
    else :
        print("Enter a valid split type")
        return

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
        "F1": f1
    })

    # Adding details to tensorboard
    writer.add_scalar(f"{split_type}/loss", running_loss / curr_count, epoch + 1)
    writer.add_scalar(f"{split_type}/accuracy", accuracy, epoch + 1)
    writer.add_scalar(f"{split_type}/balance_accuracy", balance_accuracy, epoch + 1)
    writer.add_scalar(f"{split_type}/precision", precision, epoch + 1)
    writer.add_scalar(f"{split_type}/recall", recall, epoch + 1)
    writer.add_scalar(f"{split_type}/f1", f1, epoch + 1)
    writer.add_scalar(f"{split_type}/auc", auc, epoch + 1)

    return accuracy, auc, balance_accuracy, running_loss / curr_count, f1



if __name__ == "__main__" :


    parser = argparse.ArgumentParser(description='Script for setting up training parameters.')

    # Add arguments
    parser.add_argument('--output_seq_len', type=int, default=100, help='Length of the output sequence')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=100,help='Number of training epochs')
    parser.add_argument('--embed_size', type=int, default=300, help='Size of the embedding vector')
    parser.add_argument('--dataset_type', type=str, default='dataset_2',
                        help='We have 3 options dataset_1, dataset_2, dataset_3 '
                             'dataset_2 is Original dataset ----> shivde')
    parser.add_argument("--ckpt_dir", type=str, default="/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_ckpts", help="can edit any save directory")
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
    epochs = args.epochs

    # Adding cuda device
    gpus = check_gpu_availability(2, 1, [2, 3, 4, 5, 6, 7])
    print(f"occupied {gpus}")
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{",".join(map(str, gpus))}'
    device = torch.device(f"cuda:{gpus[0]}")

    # Loading Dataset
    if args.dataset_type == "dataset_1" :
        from dataset.dataset_1 import load_dataset
        train_loader, val_loader, test_loader, num_classes, rating_counts, vectorizer, word2idx, idx2word = load_dataset(
            output_seq_len=output_seq_len, batch_size=batch_size)
    elif args.dataset_type == "dataset_2" :
        from dataset.dataset_2_shivde import load_dataset
        train_loader, val_loader, test_loader, num_classes, rating_counts, vectorizer, word2idx, idx2word = load_dataset(
            output_seq_len=output_seq_len, batch_size=batch_size)
    elif args.dataset_type == "dataset_3" :
        from dataset.dataset_3_hugging_face import load_dataset
        train_loader, val_loader, test_loader, num_classes, rating_counts, vectorizer, word2idx, idx2word = load_dataset(
            output_seq_len=output_seq_len, batch_size=batch_size)
    else :
        print("Enter valid dataset")
        exit()


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
                     pretrained_wv_type=args.pretrained_wv_type,
                     num_layers=1
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
    optimizer = torch.optim.Adam(params=model.parameters(), lr= learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max= epochs)

    # Creating exp name
    exp_name = f"model_{args.model_type}_{dataset_type}_bs_{args.batch_size}_lr_{learning_rate}_embed_{embed_size}"
    print(f"Exp name : {exp_name}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/{exp_name}')

    # Stopping Criteria
    # Parameters for early stopping
    patience = 50  # Number of epochs to wait for improvement
    min_delta = 0.001  # Minimum change to qualify as an improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Model saving
    best_val_accuracy = 0.0
    best_f1 = 0.0

    for epoch in range(epochs) :

        train_accuracy, train_auc, train_balance_accuracy, train_loss, train_f1 = train_eval_loop(epoch, model, train_loader, loss_func, optimizer,
                                                                                        scheduler, writer, "train", device)

        val_accuracy, val_auc, val_balance_accuracy, val_loss, val_f1 = train_eval_loop(epoch, model, val_loader, loss_func, optimizer,
                                                                                scheduler, writer, "val", device)

        test_accuracy, test_auc, test_balance_accuracy, test_loss, test_f1 = train_eval_loop(epoch, model, test_loader, loss_func, optimizer,
                                                                                    scheduler, writer, "test", device)

        # Model Saving
        # Check if this is the best model so far, and save it
        if val_f1 > best_f1:
            best_f1 = val_f1

            final_dict = {
                'train_accuracy' : train_accuracy,
                'train_bal_acc' : train_balance_accuracy,
                'train_auc' : train_auc,
                'train_f1' : train_f1,
                'val_accuracy' : val_accuracy,
                'val_bal_acc' : val_balance_accuracy,
                'val_auc' : val_auc,
                'val_f1' : val_f1,
                'test_accuracy' : test_accuracy,
                'test_bal_acc' : test_balance_accuracy,
                'test_auc' : test_auc,
                'test_f1' : test_f1,
                'model_weights' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'num_classes' : num_classes,
                'word2idx' : word2idx,
                'idx2word' : idx2word,
                'vectorizer' : vectorizer,
                'rating_count' : rating_counts
            }

            os.makedirs(f"{args.ckpt_dir}/{exp_name}", exist_ok=True)
            torch.save(final_dict, f"{args.ckpt_dir}/{exp_name}/{epoch}.pt")
            print(f"Best model saved with f1 score: {best_val_accuracy:.2f}%")

        # Check if validation loss improved
        if best_val_loss > val_loss :
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping check
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break