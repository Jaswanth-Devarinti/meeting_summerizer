import numpy as np
from uisrnn import UISRNN
import evals
import os
import diarization_utils
import arguments

SAVED_MODEL_NAME = os.getcwd()+'/speaker_diarization/model/saved_model.benchmark'

def test(model_args, training_args, inference_args):
  """Test pipeline.

  test model --> output result

  Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
  """

  predicted_labels = []
  test_record = []
  test_data = np.load(os.getcwd()+'/speaker_diarization/model/testing_data.npz')
  test_sequences = test_data['test_sequence']
  test_cluster_ids = test_data['test_cluster_id']
  test_sequence_list = [seq.astype(float)+0.00001 for seq in test_sequences]
  test_cluster_id_list = [np.array(cid).astype(str) for cid in test_cluster_ids]
  model = UISRNN(model_args)
  
  # testing
  model.load(SAVED_MODEL_NAME)
  for (test_sequence, test_cluster_id) in zip(test_sequence_list, test_cluster_id_list):
    predicted_label = model.predict(test_sequence, inference_args)
    predicted_labels.append(predicted_label)
    accuracy = evals.compute_sequence_match_accuracy(
        test_cluster_id, predicted_label)
    test_record.append((accuracy, len(test_cluster_id)))
    print('Ground truth labels:')
    print(test_cluster_id)
    print('Predicted labels:')
    print(predicted_label)
    print('-' * 80)

  output_string = diarization_utils.output_result(model_args, training_args, test_record)

  print('Finished diarization experiment')
  print(output_string)


def main():
  model_args, training_args, inference_args = arguments.parse_arguments()
  test(model_args, training_args, inference_args)


if __name__ == '__main__':
  main()
