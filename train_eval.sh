#!/bin/bash

# Define the parameter ranges
model_types=("lstm" "rnn" "cnn" "cnn2" "lstm2" "lstm3" "rnn2" "rnn3")
batch_sizes=(32 64 128 256 512 1024)
learning_rates=(1e-3 5e-4 1e-4 5e-5)

# Define other fixed parameters
epochs=300
dataset_type="dataset_3"

# Iterate over model_type first, then batch_size, and learning_rate
for model_type in "${model_types[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      command="python training_eval_loop.py --batch_size $batch_size --learning_rate $learning_rate --epochs $epochs --dataset_type $dataset_type --model_type $model_type"
      echo "Executing: $command"
      $command
    done
  done
done
