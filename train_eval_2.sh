#!/bin/bash

# Define the parameter ranges
model_types=("cnn2" "lstm" "lstm2" "lstm3" "gru3")
batch_sizes=(32 64 128 512)
learning_rates=(1e-3 5e-4 1e-4)

# Define other fixed parameters
epochs=200
dataset_type="dataset_2"

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
