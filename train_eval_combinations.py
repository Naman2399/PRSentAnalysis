import os
import itertools

# Define the parameter ranges
batch_sizes = [32, 64, 128, 256, 512, 1024]
learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]
model_types = ['lstm', 'rnn', 'CNN', 'cnn2', 'lstm2', 'lstm3', 'rnn2', 'rnn3']

# Define other fixed parameters
epochs = 300
dataset_type = 'dataset_3'

# Iterate over all combinations of batch_size, learning_rate, and model_type
for batch_size, learning_rate, model_type in itertools.product(batch_sizes, learning_rates, model_types):
    command = f"python training_eval_loop.py --batch_size {batch_size} --learning_rate {learning_rate} --epochs {epochs} --dataset_type {dataset_type} --model_type {model_type}"
    print(f"Executing: {command}")
    os.system(command)
