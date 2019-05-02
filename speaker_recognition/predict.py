from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
import toolkits
import preprocess
import pdb
# Parse the arguments for the speacer recognition model
import arguments_rec
args = arguments_rec.parse_arguments()

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    # Get Train/Val.    
    total_list = [os.path.join(args.data_path, file) for file in os.listdir(args.data_path)]
    unique_list = np.unique(total_list)

    # Get Model
    # construct the data generator.
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'min_slice': 720,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

    # ==> load pre-trained model ???
    if args.resume:
        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    else:
        raise IOError('==> please type in the model to load')

    print('==> start testing.')

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.
    feats = []
    for ID in unique_list:
        specs = preprocess.load_data(ID, split=False, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             min_slice=params['min_slice'])
        specs = np.expand_dims(np.expand_dims(specs[0], 0), -1)
    
        v = network_eval.predict(specs)
        feats += [v]

    feats = np.array(feats)[:,0,:]

    preprocess.similar(feats)


if __name__ == "__main__":
    main()