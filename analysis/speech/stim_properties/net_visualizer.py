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

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(a[:,0], a[:,1], "o")
# plt.figtext(a[0,0], a[0,1], classes[0])
# for i in range(len(classes)):
#     ax.text(a[i,0],a[i,1], classes[i])


#######################################
def load_models(model_dir, models = None):
    if not models:
        print("Loading models from {}".format(model_dir))
        model_fs = os.listdir(model_dir)
        model_fs = [os.path.join(model_dir, f) for f in model_fs if not ".h5" in f]
    elif models:
        print("Given {} models to load".format(len(models)))
        model_fs = [os.path.join(model_dir, f) for f in models]

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

#######################################
# Distance from activations for prototypes
def measure_distance(model_dir):

    maxact_f   = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if "class_max" in f]
    maxact_h5f = tables.open_file(maxact_f[0], mode="r+")

    # Get modelnames, first set of children under root, then load
    # hack now until make consistent way to identify models
    modelnames = [k for k in maxact_h5f.root._v_children.keys() if "conv" in k]
    models, model_fs = load_models(model_dir, modelnames)

    # Numpy arrays to store distances for each model
    n_classes = models[0].layers_by_depth[0][0].output._keras_shape[1]
    model_euc_dists = np.ndarray(shape=(len(models), n_classes, n_classes), dtype=np.float)
    model_cos_dists = np.ndarray(shape=(len(models), n_classes, n_classes), dtype=np.float)

    ## Iterate over models, stim classes, stim and get activations
    for i in trange(len(models), desc="Models", leave=True, position=0):
        # Keras function to return input to top layer
        # The K.learning_phase() function tells the model not to do dropout,
        # other shit that's only relevant for training - why we use [maxact, 0] later on
        model = models[i]
        top_activation = K.function([model.layers[0].input, K.learning_phase()],
                                    [model.layers_by_depth[0][0].input])

        #h5 group for model
        model_maxstim = maxact_h5f.root._v_children.values()[i]
        stim_classes = sorted(model_maxstim._v_children)

        # Numpy array to hold activations before distance calculation
        in_size = model.layers_by_depth[0][0].input._keras_shape[1]
        act_array = np.ndarray(shape=(in_size, len(stim_classes)), dtype=np.float)

        # Iterate over classes
        for j in trange(len(stim_classes), desc="Classes", leave=True, position=1):
            class_acts = model_maxstim._v_children[stim_classes[j]]
            act_keys = [k for k in class_acts._v_children.keys() if "maxstim" in k]

            # Numpy array to hold individual activations before averaging
            stim_act_array = np.ndarray(shape=(in_size, len(act_keys)), dtype=np.float)

            # Iterate over stimuli within a class
            for k in trange(len(act_keys), desc="Stimuli", leave=True, position=2):
                # Get stim and then activation
                max_stim = class_acts._f_get_child(act_keys[k]).read()
                stim_act_array[:,k] = top_activation([max_stim, 0])[0]

            # Save variance vector, mean variance, and take mean
            class_variance = np.var(stim_act_array, axis=1)
            mean_activation = np.mean(stim_act_array, axis=1)

            try:
                cvar_t = maxact_h5f.create_array(class_acts, "class_var", class_variance, "class variance")
                meanact_t = maxact_h5f.create_array(class_acts, "mean_act", mean_activation, "mean activation")
            except tables.NodeError:
                # we've already made these tables
                cvar_t = class_acts.class_var
                meanact_t = class_acts.mean_act
                cvar_t[:] = class_variance
                meanact_t[:] = mean_activation
            class_acts._v_attrs.mean_var = np.mean(class_variance)
            cvar_t.flush()
            meanact_t.flush()

            # add to array of all class activations
            act_array[:,j] = mean_activation

        # after getting activations from all classes, compute distances and save act_array
        euc_distances = nonzero_norm(squareform(pdist(act_array.transpose())), [0,1])
        cos_distances = squareform(pdist(act_array.transpose(), metric="cosine"))

        model_euc_dists[i, :, :] = euc_distances
        model_cos_dists[i, :, :] = cos_distances

        try:
            euc_dist_t = maxact_h5f.create_array(model_maxstim, "euc_dist", euc_distances, "euclidean distances between classes")
            cos_dist_t = maxact_h5f.create_array(model_maxstim, "cos_dist", cos_distances, "cosine distances between classes")
            act_array_t = maxact_h5f.create_array(model_maxstim, "max_acts", act_array, "maximum activations by class")
        except tables.NodeError:
            # We've already made these arrays
            euc_dist_t = model_maxstim.euc_dist
            cos_dist_t = model_maxstim.cos_dist
            act_array_t = model_maxstim.max_acts
            euc_dist_t[:,:] = euc_distances
            cos_dist_t[:,:] = cos_distances
            act_array_t[:,:] - act_array
        model_maxstim._v_attrs.class_labels = stim_classes
        euc_dist_t.flush()
        cos_dist_t.flush()
        act_array_t.flush()

    # After all models run, average, var, std for distances
    model_mean_euc = np.mean(model_euc_dists, axis=0)
    model_var_euc  = np.var(model_euc_dists, axis=0)
    model_std_euc  = np.std(model_euc_dists, axis=0)
    model_mean_cos = np.mean(model_cos_dists, axis=0)
    model_var_cos  = np.var(model_cos_dists, axis=0)
    model_std_cos  = np.std(model_cos_dists, axis=0)
    try:
        euc_t   = maxact_h5f.create_array("/", "model_euc", model_euc_dists, "euclidean distances across models")
        cos_t   = maxact_h5f.create_array("/", "model_cos", model_cos_dists, "euclidean distances across models")
        mmeuc_t = maxact_h5f.create_array("/", "model_mean_euc", model_mean_euc, "euclidean distances across models")
        mveuc_t = maxact_h5f.create_array("/", "model_var_euc", model_var_euc, "variance in euclidean distances across models")
        mseuc_t = maxact_h5f.create_array("/", "model_std_euc", model_std_euc, "std of euclidean distances across models")
        mmcos_t = maxact_h5f.create_array("/", "model_mean_cos", model_mean_cos, "cosine distances across models")
        mvcos_t = maxact_h5f.create_array("/", "model_var_cos", model_var_cos, "variance in cosine distances across models")
        mscos_t = maxact_h5f.create_array("/", "model_std_cos", model_std_cos, "std of cosine distances across models")
    except tables.NodeError:
        # We've already made these arrays
        euc_t   = maxact_h5f.root.model_euc
        cos_t   = maxact_h5f.root.model_cos
        mmeuc_t = maxact_h5f.root.model_mean_euc
        mveuc_t = maxact_h5f.root.model_var_euc
        mseuc_t = maxact_h5f.root.model_std_euc
        mmcos_t = maxact_h5f.root.model_mean_cos
        mvcos_t = maxact_h5f.root.model_var_cos
        mscos_t = maxact_h5f.root.model_std_cos
        euc_t[:,:,:] = model_euc_dists
        cos_t[:,:,:] = model_cos_dists
        mmeuc_t[:,:] = model_mean_euc
        mveuc_t[:,:] = model_var_euc
        mseuc_t[:,:] = model_std_euc
        mmcos_t[:,:] = model_mean_cos
        mvcos_t[:,:] = model_var_cos
        mscos_t[:,:] = model_std_cos

    euc_t.flush()
    cos_t.flush()
    mmeuc_t.flush()
    mveuc_t.flush()
    mseuc_t.flush()
    mmcos_t.flush()
    mvcos_t.flush()
    mscos_t.flush()

    # Save mean vars, stds as attributes
    maxact_h5f.root._v_attrs.var_euc = np.mean(model_var_euc)
    maxact_h5f.root._v_attrs.std_euc = np.mean(model_std_euc)
    maxact_h5f.root._v_attrs.var_cos = np.mean(model_var_cos)
    maxact_h5f.root._v_attrs.std_cos = np.mean(model_std_cos)

    maxact_h5f.close()


def usage():
    print("Tools for analysis and visualization of trained networks")
    print("    -h, --help           : print this message")
    print("    -m, --mode <string>  : operating mode")
    print("        available modes:")
    print("            \"maxact\"   : generate stim that maximally activate classes")
    print("            \"distance\" : compute distances between stimulus classes")
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

        # print("Maximum activations calculated, compute distances?")

    elif mode == "distance":
        print("Measuring distances between stimulus classes across models")

        measure_distance(model_dir)


    # Sne weights on last layer
    #positions, distances = sne_last_weights(models)








