import os

# Parse the argument
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser()
    # set up training configuration.
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--resume', default=os.getcwd()+'/speaker_recognition/model/weights.h5', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--data_path', default=os.getcwd()+'/speaker_recognition/test_features', type=str)
    # set up network configuration.
    parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
    parser.add_argument('--ghost_cluster', default=2, type=int)
    parser.add_argument('--vlad_cluster', default=8, type=int)
    parser.add_argument('--bottleneck_dim', default=512, type=int)
    parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'gvlad'], type=str)
    # set up learning rate, training loss and optimizer.
    parser.add_argument('--epochs', default=56, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--warmup_ratio', default=0, type=float)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)
    parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
    parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
    parser.add_argument('--ohem_level', default=0, type=int, help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
    # set up path to the wav file which should be derialized
    parser.add_argument('--diar', default=os.getcwd()+'/test_wavs/Australian.wav', type=str)

    args = parser.parse_args()
    return args
