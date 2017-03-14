import os
from keras.models import load_model
from sklearn.manifold import TSNE
import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import tables

from keras import backend as K

# Call training sample for sizes, etc.
X_train,cons_train,speak_train,vow_train = spectrogram_for_training(file_loc, MEL_PARAMS,phovect_idx,N_JIT,JIT_AMT)


# sign in to plotly
plotly.tools.set_credentials_file(username="crimedude", api_key="xWT0oY7xCuFRItFSmTkP")

# load models
model_dir = "/home/lab/Speech_Models/conscat"
model_fs = os.listdir(model_dir)
model_fs = [os.path.join(model_dir,f) for f in model_fs]
models = [load_model(f) for f in model_fs]

# Sne weights on last layer
positions, distances = sne_last_weights(models)

# scatterplot positions
scat_traces = scatter_positions(positions)
py.plot(scat_traces, filename="scatter-positions3")

#######################################
# Max activations of classes

## Params
n_stim   = 5 # num stim per model/class
classind = 2 # /b/ of speaker 3

classes = ['1g','2g','3g','4g','5g','1b','2b','3b','4b','5b']

img_shape = X_train.shape[1:3]
min_max = [np.min(X_train), np.max(X_train)]

## Dataframe to store
h5f = tables.open_file(model_dir+"/class_max_2.h5", mode="w", title = "Max class of classifiers")



for j in range(len(models)):
    modelname = model_fs[j].split('/')[-1]
    group = h5f.create_group("/", modelname, "Max stim for model")
    for i in range(10):
        stimclass = h5f.create_group(group, classes[i], "Max stim for class {}".format(classes[i]))
        for k in range(n_stim):
            print("model: {}, class: {}, iter: {}".format(str(j), classes[i], str(k)))
            preds, img = max_class_act(models[j], i, img_shape, min_max)
            img_t = h5f.create_array(stimclass, classes[i]+"_"+str(k)+"_maxstim", img, "MaxStim")
            preds_t = h5f.create_array(stimclass, classes[i]+"_"+str(k)+"_preds", preds, "Preds")
h5f.close()




#######################################
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


def max_class_act(model, classind, img_shape, min_max, min_acc=0.995):
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
    # run gradient ascent for 20 steps
    step = 10000
    while True:
        loss_value, grads_value = iterate([input_img_data, 0])
        input_img_data += grads_value * step
        if loss_value >= min_acc:
            break
        else:
            print(loss_value)

    # rescale to orig scale
    rs_factor = min_max[1] / np.max(input_img_data)
    input_img_data = input_img_data * rs_factor

    preds = model.predict(input_img_data)
    return preds, input_img_data
