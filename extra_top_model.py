import os
import csv
from tensorboard.backend.event_processing import event_accumulator


def extract_top_checkpoints(model_log_dir):
    # Function to extract the checkpoint with the highest validation and test accuracy and F1 scores
    top_val_acc_epoch = None
    top_test_acc_epoch = None
    top_val_f1_epoch = None
    top_test_f1_epoch = None

    highest_val_acc = -float('inf')
    highest_test_acc = -float('inf')
    highest_val_f1 = -float('inf')
    highest_test_f1 = -float('inf')

    for event_file in os.listdir(model_log_dir):
        event_file_path = os.path.join(model_log_dir, event_file)
        ea = event_accumulator.EventAccumulator(event_file_path)
        ea.Reload()

        if 'val/accuracy' in ea.Tags()['scalars']:
            val_scalars = ea.Scalars('val/accuracy')
            max_val_acc_entry = max(val_scalars, key=lambda x: x.value)
            val_acc = max_val_acc_entry.value
            if val_acc > highest_val_acc:
                highest_val_acc = val_acc
                top_val_acc_epoch = max_val_acc_entry.step

        if 'test/accuracy' in ea.Tags()['scalars']:
            test_scalars = ea.Scalars('test/accuracy')
            max_test_acc_entry = max(test_scalars, key=lambda x: x.value)
            test_acc = max_test_acc_entry.value
            if test_acc > highest_test_acc:
                highest_test_acc = test_acc
                top_test_acc_epoch = max_test_acc_entry.step

        if 'val/f1' in ea.Tags()['scalars']:
            val_f1_scalars = ea.Scalars('val/f1')
            max_val_f1_entry = max(val_f1_scalars, key=lambda x: x.value)
            val_f1 = max_val_f1_entry.value
            if val_f1 > highest_val_f1:
                highest_val_f1 = val_f1
                top_val_f1_epoch = max_val_f1_entry.step

        if 'test/f1' in ea.Tags()['scalars']:
            test_f1_scalars = ea.Scalars('test/f1')
            max_test_f1_entry = max(test_f1_scalars, key=lambda x: x.value)
            test_f1 = max_test_f1_entry.value
            if test_f1 > highest_test_f1:
                highest_test_f1 = test_f1
                top_test_f1_epoch = max_test_f1_entry.step

    return {
        'val_acc_epoch': top_val_acc_epoch,
        'val_acc': highest_val_acc,
        'test_acc_epoch': top_test_acc_epoch,
        'test_acc': highest_test_acc,
        'val_f1_epoch': top_val_f1_epoch,
        'val_f1': highest_val_f1,
        'test_f1_epoch': top_test_f1_epoch,
        'test_f1': highest_test_f1
    }


def find_best_checkpoints(root_log_dir, top_n=20):
    model_accuracies = []

    # Iterate over each model directory
    for model_name in os.listdir(root_log_dir):
        model_log_dir = os.path.join(root_log_dir, model_name)
        if os.path.isdir(model_log_dir):
            metrics = extract_top_checkpoints(model_log_dir)
            if metrics['val_acc_epoch'] and metrics['test_acc_epoch']:
                model_accuracies.append((model_name, metrics))

    # Sort the models based on validation accuracy (primary) and test accuracy (secondary)
    model_accuracies.sort(key=lambda x: (-x[1]['val_acc'], -x[1]['test_acc']))

    # Return the top N models
    return model_accuracies[:top_n]


def save_results_to_csv(results, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model Name', 'Val Accuracy', 'Val Acc Epoch', 'Test Accuracy', 'Test Acc Epoch',
                         'Val F1', 'Val F1 Epoch', 'Test F1', 'Test F1 Epoch'])
        for model_name, metrics in results:
            writer.writerow([model_name,
                             metrics['val_acc'], metrics['val_acc_epoch'],
                             metrics['test_acc'], metrics['test_acc_epoch'],
                             metrics['val_f1'], metrics['val_f1_epoch'],
                             metrics['test_f1'], metrics['test_f1_epoch']])


if __name__ == "__main__":
    root_log_dir = '/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/runs/dataset_2'  # Replace with your path
    output_csv = '/raid/home/namanmalpani/final_yr/DLNLP_Assignment_1/best_checkpoints_dataset_2.csv'  # Replace with your desired output path

    best_checkpoints = find_best_checkpoints(root_log_dir)
    save_results_to_csv(best_checkpoints, output_csv)

    print(f"Results saved to {output_csv}")

    # Model checkpoint directory
    model_ckpt_dir = "/data/home/karmpatel/karm_8T/naman/demo/DLNLP_Ass1_Data/model_ckpts"
    model_types = ["lstm", "lstm2", "lstm3", "rnn" , "rnn2" , "rnn3" , "cnn" , "cnn2", "gru", "gru2", "gru3"]
    model_ckpts = {}
    for model_name, metrics in best_checkpoints:
        val_ckpt_path = os.path.join(model_ckpt_dir, model_name, f"{metrics['val_acc_epoch']}.pt")
        test_ckpt_path = os.path.join(model_ckpt_dir, model_name, f"{metrics['test_acc_epoch']}.pt")
        print(f"Model: {model_name}")
        print(f"  - Validation Checkpoint Path: {val_ckpt_path}")
        print(f"  - Test Checkpoint Path: {test_ckpt_path}")
        for model_type in model_types :
            if f"model_{model_type}_" in model_name :
                if model_type not in model_ckpts.keys() :
                    model_ckpts[model_type] = [val_ckpt_path]
                else :
                    model_ckpts[model_type].append(val_ckpt_path)

    print("-" * 50)
    print("Model checkpoint dict")
    for key, value in model_ckpts.items():
        print(f"Key: {key}, Value: {value}")

    print("-" * 50)
    print(model_ckpts)


