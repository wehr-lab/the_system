from keras.models import load_model
from sklearn.manifold import TSNE
import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# sign in to plotly
plotly.tools.set_credentials_file(username="crimedude", api_key="xWT0oY7xCuFRItFSmTkP")

# load yer model
model = load_model("/home/lab/Speech_Models/multi_mel_conv_E99-L0.46-cons0.97_speak0.99_vow0.98")

# First try - t-sne the weights at the last output layer for speakers
speak_layer = model.layers_by_depth[0][1]
speak_layer_weights = speak_layer.get_weights()[0]
speak_layer_weights = speak_layer_weights.transpose()


sne = TSNE(n_components=1, learning_rate=100, verbose=2)
spk_sne = sne.fit_transform(speak_layer_weights)

trace = go.Scatter(
    x=spk_sne[:,0],
    y=spk_sne[:,1],
    text=["1","2","3","4","5"],
    mode='markers+text')
data = [trace]
py.plot(data, filename='scatter-test')