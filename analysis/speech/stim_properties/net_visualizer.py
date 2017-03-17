from __future__ import print_function
import os
import sys
import getopt
import warnings
sys.path.append("/home/lab/github/the_system")

from time import strftime, gmtime
from keras.models import load_model
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial.distance import pdist, squareform
import tables
from collections import OrderedDict as odict
from tqdm import trange, tqdm

from keras import backend as K

from learning_utils import *
from params import Learn_params









#######################################
# Distance from activations for prototypes
# h5f = tables.open_file(model_dir+"/class_max_2.h5", mode="r")
# model = models[0]
#
# activation_out = K.function([model.layers[0].input, K.learning_phase()],
#                             [model.layers_by_depth[0][0].input])
#
# model_maxstim = h5f.root._v_children.values()[0]
# sample_maxstim = list()
# for i in range(n_stim):
#     sample_maxstim.append(odict())
#     for stimclass in sorted(model_maxstim._v_children):
#         acts = model_maxstim._v_children[stimclass]
#         child_keys = [k for k in acts._v_children.keys() if "maxstim" in k]
#         child_key = child_keys[i]
#         sample_maxstim[i][stimclass] = acts._f_get_child(child_key).read()
#
# # Get sample output for shape
# n_col = np.ndarray.flatten(activation_out([sample_maxstim[0]['1b'], 0])[0]).shape[0]
# n_row = len(sample_maxstim[0])
#
# # Get out weights and reorder
# out_weights = model.layers[-1].get_weights()[0].transpose()
# # we need to take [1g,2g,3g,4g,5g,1b,2b,3b,4b,5b] to [1b,1g,2b,2g,3b,3g,4b,4g,5b,5g]
# row_order = [5,0,6,1,7,2,8,3,9,4]
# out_weights = out_weights[row_order,:]
#
# # want a dictionary of dict['input_stimclass'] = distances, so a distance matrix for every stimclass
# model_distances = list()
# for i in range(n_stim):
#     dist_dict = odict()
#     for stimclass, maxact in sample_maxstim[i].items():
#         this_act = activation_out([maxact, 0])[0]
#         class_act = np.multiply(this_act, out_weights)
#         dist_dict[stimclass] = nonzero_norm(squareform(pdist(class_act)), [0,1])
#     model_distances.append(dist_dict)
#
#
#
#
#
#     model_acts = np.multiply(model_acts, out_weights)
#     model_distances.append(nonzero_norm(squareform(pdist(model_acts)), [0,1]))

#######################################
def load_models(model_dir):
    print("Loading models from {}".format(model_dir))
    model_fs = os.listdir(model_dir)
    model_fs = [os.path.join(model_dir, f) for f in model_fs if not ".h5" in f]

    models = list()
    for f in tqdm(model_fs):
        models.append(load_model(f))
    print("{} models loaded".format(len(models)))
    return models, model_fs

def nonzero_norm(X, range=[0,1]):
    zeroinds = np.nonzero(X)
    X[zeroinds] = (X[zeroinds]-np.min(X[zeroinds]))/(np.max(X[zeroinds])-np.min(X[zeroinds]))
    X[zeroinds] = X[zeroinds] * (range[1]-range[0]) + range[0]
    return X

def sne_last_weights(models):
    # Embed last layer weights in 2D and get distance matrix
    # Input: model or list of models

    if type(models) is not list:
        models = [models]

    n_models = len(models)
    # top layer, first net of the layer, # output weights
    n_classes = len(models[0].layers_by_depth[0][0].get_weights()[1])

    # prepare arrays
    # distances (n_classes x n_classes x n_models)
    distances = np.zeros((n_classes, n_classes, n_models), dtype=np.float)
    positions = np.zeros((n_classes, 2, n_models), dtype=np.float)

    # make t-sne object
    sne = TSNE(n_components=2, learning_rate=100)

    for i, m in enumerate(models):
        l_weights = m.layers_by_depth[0][0].get_weights()[0].transpose()
        spk_sne = sne.fit_transform(l_weights)
        positions[:,:,i] = spk_sne
        distances[:,:,i] = squareform(pdist(spk_sne))

    return positions, distances

def scatter_positions(positions):
    traces = []
    for i in range(positions.shape[2]):
        traces.append(go.Scatter(
            x = positions[:,0,i],
            y = positions[:,1,i],
            text=["1b","2b","3b","4b","5b","1g","2g","3g","4g","5g"],
            mode="markers+text"))

    return traces

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#######################################
# Max activations of classes

def generate_class_act(model, classind, img_shape, min_max, min_acc=0.995):
    # Generate a "stimulus" that maximizes the activation of one of the numbered class

    model_in = model.layers[0].input
    model_out = model.layers[-1].output

    # layer_output = K.function([model.layers[0].input, K.learning_phase()],
    #                          [model.layers[-1].output])

    loss = model_out[0, classind]

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model_in)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([model_in, K.learning_phase()], [loss, grads])

    # we start from a gray image with some noise
    input_img_data = np.random.uniform(low=min_max[0], high=min_max[1],
                                       size=(1, img_shape[0], img_shape[1], 1))
    # run gradient ascent
    step = 10000

    t = tqdm(total=min_acc, desc="Activation", position=3)
    last_loss = 0
    while True:
        loss_value, grads_value = iterate([input_img_data, 0])
        input_img_data += grads_value * step
        if loss_value >= min_acc:
            break
        else:
            t.update(loss_value-last_loss)
            last_loss = loss_value
    t.close()

    # rescale to orig scale
    rs_factor = min_max[1] / np.max(input_img_data)
    input_img_data = input_img_data * rs_factor

    preds = model.predict(input_img_data)
    return preds, input_img_data

def max_models_classes(model_dir, n_stim, class_labels, img_shape, min_max):
    """Wrapper to generate artificial stimuli that maximally activate model classes

    Args:
        model_dir (str): dir containing keras models to load, location of saved h5 file
        n_stim (int): number of stim to generate per model per class
        class_labels (list): list of strings used to label classes. should correspond to order of output neurons
        img_shape (tuple): tuple that describes shape of input
        min_max (list): list of two values (int or float) to rescale the generated stim to same range as training stim

    Returns:
        h5_loc (string): string for location of
    """
    # Filter annoying warnings from tables
    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

    # Load models
    models, model_fs = load_models(model_dir)

    ## Dataframe to store
    filename = model_dir+"/class_max_{}.h5".format(strftime("%y%m%d-%H%M",gmtime()))
    h5f = tables.open_file(filename, mode="w", title = "Max class of classifiers")

    print("-------------------")
    print("Generating Stimuli")
    # Iterate over models...
    for j in trange(len(models), desc="Models", leave=True, position=0):
        modelname = model_fs[j].split('/')[-1]
        group = h5f.create_group("/", modelname, "Max stim for model")

        # and classes
        for i in trange(len(class_labels), desc="Classes", leave=True, position=1):
            stimclass = h5f.create_group(group, class_labels[i], "Max stim for class {}".format(class_labels[i]))

            # to generate n stim
            for k in trange(n_stim, desc="Stimuli", leave=True, position=2):
                # print("model: {}, class: {}, iter: {}".format(str(j), classes[i], str(k)), end="\r")
                preds, img = generate_class_act(models[j], i, img_shape, min_max)
                img_t = h5f.create_array(stimclass, class_labels[i]+"_"+str(k)+"_maxstim", img, "MaxStim")
                preds_t = h5f.create_array(stimclass, class_labels[i]+"_"+str(k)+"_preds", preds, "Preds")
                img_t.flush()
                preds_t.flush()
    h5f.close()

    print("Max stim saved to {}".format(filename))

def usage():
    print("Tools for analysis and visualization of trained networks")
    print("    -h, --help           : print this message")
    print("    -m, --mode <string>  : operating mode")
    print("        available modes:")
    print("            \"maxact\": generate stim that maximally activate classes")
    print("    -p, --path <string>  : path to load models from")
    print("    -n, --nstim <int>    : number of stimuli to generate per model per class")
    print("    -c, --classes <list> : list of strings for class labels")


#################################
# Script running stuff

if __name__ == "__main__":
    try:
        opts,args = getopt.getopt(sys.argv[1:],"hm:p:n:c:", ["help", "mode", "path", "nstim", "classes"])
    except:
        opts = None
        args = None

    # load defaults first and then change if requested by options
    model_dir = Learn_params.MODEL_DIR
    n_stim = Learn_params.N_STIM
    stim_classes = Learn_params.CLASSES

    # parse cmd options
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt in ('-m', '--mode'):
            mode = str(arg)
            print("Running in mode: {}".format(arg))
        elif opt in ('-p', '--path'):
            print('Path set to {}'.format(arg))
            model_dir = arg
        elif opt in ('-n', '--nstim'):
            try:
                n_stim = int(arg)
            except:
                TypeError("n_stim must be integer")
            print('Number of stim set to {}'.format(arg))
        elif opt in ('-c', '--classes'):
            stim_classes = arg
            print("Stim classes set to {}".format(arg))

    if not mode:
        RuntimeError("mode is unset! pass an argument to -m, use -h for help")

    if mode == "maxact":
        print("Loading initial phoneme iterators")
        phovect_idx, phovect_xdi, file_idx, file_loc = (
            make_phoneme_iterators(Learn_params.NAMES, Learn_params.CONS,
                                   Learn_params.VOWS, Learn_params.MAPBACK,
                                   Learn_params.PHONEME_DIR))
        X_train, cons_train, speak_train, vow_train = (
            spectrogram_for_training(file_loc, Learn_params.MEL_PARAMS,
                                     phovect_idx, Learn_params.N_JIT,
                                     Learn_params.JIT_AMT))

        img_shape = X_train.shape[1:3]
        min_max = [np.min(X_train), np.max(X_train)]

        max_models_classes(model_dir, n_stim, stim_classes, img_shape, min_max)



    # Sne weights on last layer
    #positions, distances = sne_last_weights(models)








