import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
def initialize_GPU(args):
    # Initialize GPUs
    #import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def get_chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def debug_generator(generator):
    import cv2
    import pdb
    G = generator.next()
    for i,img in enumerate(G[0]):
        path = '../sample/{}.jpg'.format(i)
        img = np.asarray(img[:,:,::-1] + 128.0, dtype='uint8')
        cv2.imwrite(path, img)


# set up multiprocessing
def set_mp(processes=8):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool

def get_voxceleb2_datalist(args, path):
    with open(path) as f:
        strings = f.readlines()
        audiolist = np.array([os.path.join(args.data_path, string.split()[0]) for string in strings])
        labellist = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return audiolist, labellist


def calculate_eer(y, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def sync_model(src_model, tgt_model):
    print('==> synchronizing the model weights.')
    params = {}
    for l in src_model.layers:
        params['{}'.format(l.name)] = l.get_weights()

    for l in tgt_model.layers:
        if len(l.get_weights()) > 0:
            l.set_weights(params['{}'.format(l.name)])
    return tgt_model