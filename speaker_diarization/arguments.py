"""Arguments"""
import argparse
import os

def parse_arguments():
  """Parse arguments.
  Returns:
    A tuple of:

      - model_args: model arguments
      - training_args: training arguments
      - inference_args: inference arguments
  """
  # model configurations
  model_parser = argparse.ArgumentParser(
      description='Model configurations.', add_help=False)

  model_parser.add_argument(
      '--diar',
      default=os.getcwd()+'/test_wavs/Australian.wav',
      type=str,
      help='Path to the wav file which should be derialized.')

  model_parser.add_argument(
      '--observation_dim',
      default=512,
      type=int,
      help='The dimension of the embeddings (e.g. d-vectors).')

  model_parser.add_argument(
      '--rnn_hidden_size',
      default=512,
      type=int,
      help='The number of nodes for each RNN layer.')
  model_parser.add_argument(
      '--rnn_depth',
      default=1,
      type=int,
      help='The number of RNN layers.')
  model_parser.add_argument(
      '--rnn_dropout',
      default=0.2,
      type=float,
      help='The dropout rate for all RNN layers.')
  model_parser.add_argument(
      '--transition_bias',
      default=None,
      type=float,
      help='If the value is given, it will fix to this value. If the '
           'value is None, it will estimate it from training data '
           'using MLE Estimation.')
  model_parser.add_argument(
      '--crp_alpha',
      default=1.0,
      type=float,
      help='The value of alpha for the Chinese restaurant process (CRP).'
           'In this implementation, currently it only'
           'supports using a given value of crp_alpha.')
  model_parser.add_argument(
      '--sigma2',
      default=None,
      type=float,
      help='The value of sigma squared, If the value is given, it will fix to this value. If the '
           'value is None, it will estimate it from training data.')
  model_parser.add_argument(
      '--verbosity',
      default=2,
      type=int,
      help='How verbose will the logging information be. Higher value '
      'represents more verbose information. A general guideline: '
      '0 for errors; 1 for finishing important steps; '
      '2 for finishing less important steps; 3 or above for debugging '
      'information.')

  # training configurations
  training_parser = argparse.ArgumentParser(
      description='Training configurations.', add_help=False)

  training_parser.add_argument(
      '--optimizer',
      '-o',
      default='adam',
      choices=['adam'],
      help='The optimizer for training.')
  training_parser.add_argument(
      '--learning_rate',
      '-l',
      default=1e-4,
      type=float,
      help='The leaning rate for training.')
  training_parser.add_argument(
      '--learning_rate_half_life',
      '-hl',
      default=1000,
      type=int,
      help='The half life of the leaning rate for training. If this value is '
           'positive, it reduces learning rate by half every this many '
           'iterations during training. If this value is 0 or negative, '
           'it does not decay learning rate.')
  training_parser.add_argument(
      '--train_iteration',
      '-t',
      default=3000,
      type=int,
      help='The total number of training iterations.')
  training_parser.add_argument(
      '--batch_size',
      '-b',
      default=30,
      type=int,
      help='The batch size for training.')
  training_parser.add_argument(
      '--num_permutations',
      default=20,
      type=int,
      help='The number of permutations per utterance sampled in the training '
           'data.')
  training_parser.add_argument(
      '--sigma_alpha',
      default=1.0,
      type=float,
      help='The inverse gamma shape for estimating sigma2. This value is only '
           'meaningful when sigma2 is not given, and estimated from data.')
  training_parser.add_argument(
      '--sigma_beta',
      default=1.0,
      type=float,
      help='The inverse gamma scale for estimating sigma2. This value is only '
           'meaningful when sigma2 is not given, and estimated from data.')
  training_parser.add_argument(
      '--regularization_weight',
      '-r',
      default=1e-5,
      type=float,
      help='The network regularization multiplicative.')
  training_parser.add_argument(
      '--grad_max_norm',
      default=5.0,
      type=float,
      help='Max norm of the gradient.')
  training_parser.add_argument(
      '--enforce_cluster_id_uniqueness',
      default=False,
      type=bool,
      help='Whether to enforce cluster ID uniqueness across different '
           'training sequences. Only effective when the first input to fit() '
           'is a list of sequences. In general, assume the cluster IDs for two '
           'sequences are [a, b] and [a, c]. If the `a` from the two sequences '
           'are not the same label, then this arg should be True.')

  # inference configurations
  inference_parser = argparse.ArgumentParser(
      description='Inference configurations.', add_help=False)

  inference_parser.add_argument(
      '--beam_size',
      '-s',
      default=10,
      type=int,
      help='The beam search size for inference.')
  inference_parser.add_argument(
      '--look_ahead',
      default=1,
      type=int,
      help='The number of look ahead steps during inference.')
  inference_parser.add_argument(
      '--test_iteration',
      default=2,
      type=int,
      help='During inference, it concatenates M duplicates of the test '
           'sequence, and run inference on this concatenated sequence. '
           'Then it returns the inference results on the last duplicate as the '
           'final prediction for the test sequence.')

  # a super parser for sanity checks
  super_parser = argparse.ArgumentParser(
      parents=[model_parser, training_parser, inference_parser])

  # get arguments
  super_parser.parse_args()
  model_args, _ = model_parser.parse_known_args()
  training_args, _ = training_parser.parse_known_args()
  inference_args, _ = inference_parser.parse_known_args()

  return (model_args, training_args, inference_args)
