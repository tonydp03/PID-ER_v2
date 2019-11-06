import sys
import tensorflow as tf
import keras
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from keras import optimizers
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from keras import backend as K
from tqdm import tqdm


# reserve only the 30% of the GPU memory
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


batch_size = 512
width = 50
height = 10
channels = 3
classes = 6
epochs = 100
dataset_dir = '/lustrehome/adipilato/ParticleID/new_datasets/5PartPerEvent/padded/train/'
save_dir = '/lustrehome/adipilato/ParticleID/PID-ER_v2/models/'
padding = 'padding' + str(height)
model_name= padding + '_ATmodelV2'
history_name = model_name + '_history'

class_labels = {22:0, -11:1, 11:1, -13:2, 13:2, -211:3, 211:3, 311:4, -1:5, -9999:5}



# arrays of data needed for training

data_list = []
pid_list = []
en_list = []

# read dataset
files = [f for f in os.listdir(dataset_dir) if f.endswith("h5")]

for name in tqdm(files):
    print("Reading file", name)
    data = pd.read_hdf(dataset_dir + name)
    num_tracks = data.trackster.max()
    print(num_tracks)

    tracksters = np.array([data.E.values, data.eta.values, data.phi.values]).T.reshape(-1, width, height, channels)

    pid_vals = np.array([data.pid.values]).T.reshape(-1, width*height)
    select = (np.max(pid_vals,axis=1)==0)
    pid_vals=np.max(np.abs(pid_vals),axis=1)
    pid_vals[select]=pid_vals[select]*-1.0
    pid_vals = pid_vals.tolist()
    pid_arr = [class_labels[x] for x in pid_vals]
    en_vals = np.array([data.genE.values]).T.reshape(-1, width*height)
    en_arr = np.max(en_vals,axis=1)

    data_list.append(tracksters)
    pid_list.append(pid_arr)
    en_list.append(en_arr)

data_array = np.array([item for sublist in data_list for item in sublist])
print(data_array.shape)

pid_array = np.array([item for sublist in pid_list for item in sublist])
pid_array = keras.utils.to_categorical(pid_array, num_classes=classes, dtype='float32')
print(pid_array.shape)

en_array = en_array = np.array([item for sublist in en_list for item in sublist])
print(en_array.shape)

mean_en = np.mean(en_array)
std_en = np.std(en_array)
print('Mean Energy Value: {}'.format(mean_en))
print('Std Energy Value: {}'.format(std_en))

en_array_norm = (en_array - mean_en)/std_en


print('Creating model...')

def full_model():

    input_img = Input(shape=(width, height, channels), name='input')
    
    conv = Conv2D(16, (5,1), activation='relu', padding='same', kernel_initializer='random_uniform', data_format='channels_last', name='conv1')(input_img)
    conv = Conv2D(16, (3,3), activation='relu', padding='same', kernel_initializer='random_uniform', data_format='channels_last', name='conv2')(conv)
    conv = Conv2D(16, (3,3), activation='relu', padding='same', kernel_initializer='random_uniform', data_format='channels_last', name='conv3')(conv)

    flat = Flatten()(conv)

    dense = Dense(512, activation='relu', kernel_initializer='random_uniform', name='dense1')(flat)
#     drop = Dropout(0.4)(dense)
    dense = Dense(128, activation='relu', kernel_initializer='random_uniform', name='dense2')(dense)
#     drop = Dropout(0.5)(dense)

    dense_id = Dense(64, activation='relu', kernel_initializer='random_uniform', name='dense_id1')(dense)
#     drop_id = Dropout(0.5)(dense_id)
    dense_id = Dense(16, activation='relu', kernel_initializer='random_uniform', name='dense_id2')(dense_id)
#     drop_id = Dropout(0.5)(dense_id)

    pid = Dense(classes, activation='softmax', kernel_initializer='random_uniform', name='pid_output')(dense_id)

    dense_er = Dense(64, activation='relu', kernel_initializer='random_uniform', name='dense_er1')(dense)
    dense_er = Dense(16, activation='relu', kernel_initializer='random_uniform', name='dense_er2')(dense_er)
    
    enreg = Dense(1, name='enreg_output')(dense_er)

    model = Model(inputs=input_img, outputs=[pid, enreg])

    model.compile(loss={'pid_output': 'categorical_crossentropy', 'enreg_output': 'mse'}, loss_weights={'pid_output': 1, 'enreg_output': 1}, optimizer='adam', metrics={'pid_output': 'accuracy', 'enreg_output': 'mse'})
    return model

model = full_model()
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

print('Frozen model saved')

print("Done!")



