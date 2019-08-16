# coding: utf-8

import sys
import tensorflow as tf
import keras
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers
import os
import numpy as np
import pandas as pd

from keras import backend as K
from tqdm import tqdm


batch_size = 256
width = 50
height = 10
channels = 3
classes = 4
epochs = 100
dataset_dir = '/data/user/adipilat/ParticleID/genEvts/'
save_dir = '/data/user/adipilat/ParticleID/models/'
padding = 'padding' + str(height)
model_name= padding +'_model'
history_name = padding + '_history'

# This dictionary should be extended to new classes and antiparticles
class_labels = {22:0, 11:1, 13:2, 211:3}

# arrays of data needed for training
data_array = []
pid_array = []
en_array = []

# read dataset
files = [f for f in os.listdir(dataset_dir) if f.endswith("h5")]

for name in tqdm(files):
    print("Reading file", name)
    data = pd.read_hdf(dataset_dir + name)
    n_events = int(0.9 * data.event.max()) # using 90% of events for training

    for i in range(1, int(n_events+1)):
        tracksters = data.loc[(data['event'] == float(i)) & (data['trackster'] != float(0))]
        n_tracksters = tracksters.trackster.max()
        lead_en = 0
        for j in range(1, int(n_tracksters)+1):
            layerclusters = tracksters.loc[tracksters['trackster'] == float(j)]
            en = np.sum(layerclusters["E"].values)
            if(en>lead_en): #### since I shooted a single particle, only the leading trackster is considered
                lead_en = en
                image = np.zeros(width*height*channels).reshape(width,height,channels)
                pid = int(layerclusters["pid"].iloc[0])
                pid = class_labels[pid]
                en_value = layerclusters["genE"].iloc[0]
                for k in range(1, width+1):
                    layer = layerclusters[layerclusters['layer'] == float(k)]
                    if(len(layer) != 0):
                        temp = layer.E.values, layer.eta.values, layer.phi.values
                        temp = np.array(temp).T
                        dim = min(temp.shape[0],height)
                        image[k-1][:dim] = temp[:dim]
        data_array.append(image)
        pid_array.append(pid)
        en_array.append(en_value)    

    print("File", name, " processed")

data_array = np.array(data_array)
pid_array = np.array(pid_array)
pid_array = keras.utils.to_categorical(pid_array, num_classes=classes, dtype='float32')
en_array = np.array(en_array)

print(data_array.shape)
print(en_array.shape)
print(pid_array.shape)

####### NORMALIZE THE ENERGY ########
mean_en = np.mean(en_array)
std_en = np.std(en_array)
print('Mean Energy Value: {}'.format(mean_en))
print('Std Energy Value: {}'.format(std_en))

en_array_norm = (en_array - mean_en)/std_en

print('Creating model...')

def tree_model():

    input_img = Input(shape=(width, height, channels), name='input')
    
    conv = Conv2D(3, (5,1), activation='relu', padding='same', kernel_initializer='random_uniform', data_format='channels_last', name='conv1')(input_img)
    conv = Conv2D(3, (3,3), activation='relu', padding='same', kernel_initializer='random_uniform', data_format='channels_last', name='conv2')(conv)
    conv = Conv2D(3, (3,3), activation='relu', padding='same', kernel_initializer='random_uniform', data_format='channels_last', name='conv3')(conv)

    flat = Flatten()(conv)

    dense = Dense(1024, activation='relu', kernel_initializer='random_uniform', name='dense1')(flat)
    dense = Dense(256, activation='relu', kernel_initializer='random_uniform', name='dense2')(dense)

    dense_id = Dense(64, activation='relu', kernel_initializer='random_uniform', name='dense_id1')(dense)
    dense_id = Dense(16, activation='relu', kernel_initializer='random_uniform', name='dense_id2')(dense_id)
    pid = Dense(classes, activation='softmax', kernel_initializer='random_uniform', name='pid_output')(dense_id)

    dense_er = Dense(64, activation='relu', kernel_initializer='random_uniform', name='dense_er1')(dense)
    dense_er = Dense(8, activation='relu', kernel_initializer='random_uniform', name='dense_er2')(dense_er)
    enreg = Dense(1, name='enreg_output')(dense_er)

    model = Model(inputs=input_img, outputs=[pid, enreg])

    model.compile(loss={'pid_output': 'categorical_crossentropy', 'enreg_output': 'mse'}, loss_weights={'pid_output': 1, 'enreg_output': 2}, optimizer='adam', metrics={'pid_output': 'accuracy', 'enreg_output': 'mse'})
    return model

model = tree_model()
model.summary()

print("Model created!")

history = model.fit(data_array, {'pid_output': pid_array, 'enreg_output': en_array_norm}, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
history_save = pd.DataFrame(history.history).to_hdf(save_dir + history_name + ".h5", "history", append=False)

# Save model and weights
model.save(save_dir + model_name + ".h5")
print('Saved trained model at %s ' % save_dir)

# save the frozen model
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, save_dir, model_name + ".pbtxt", as_text=True)
tf.train.write_graph(frozen_graph, save_dir, model_name + ".pb", as_text=False)
print('Model saved')

print("Done!")

