import numpy as np
from uisrnn import UISRNN
import os
import arguments

SAVED_MODEL_NAME = os.getcwd()+'/speaker_diarization/model/saved_model.benchmark'

def train(model_args, training_args):
  """Train pipeline.

  Load data --> train model

  Args:
    model_args: model configurations
    training_args: training configurations
  """

  predicted_labels = []
  test_record = []
  train_data = np.load(os.getcwd()+'/speaker_diarization/model/training_data.npz')
  train_sequence = train_data['train_sequence']
  train_cluster_id = train_data['train_cluster_id']
  train_sequence_list = [seq.astype(float)+0.00001 for seq in train_sequence]
  train_cluster_id_list = [np.array(cid).astype(str) for cid in train_cluster_id]
  model = UISRNN(model_args)
  
  # training
  model.fit(train_sequence_list, train_cluster_id_list, training_args)
  model.save(SAVED_MODEL_NAME)

def main():
  model_args, training_args, _ = arguments.parse_arguments()
  train(model_args, training_args)

if __name__ == '__main__':
  main()
