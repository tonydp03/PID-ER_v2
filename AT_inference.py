#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K

from sklearn.metrics import confusion_matrix
import itertools

from tqdm import tqdm

# reserve only the 30% of the GPU memory
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


# In[2]:


width = 50
height = 10
channels = 3
classes = 6
dataset_dir = '/lustrehome/adipilato/ParticleID/new_datasets/5PartPerEvent/padded/test/'
save_dir = '/lustrehome/adipilato/ParticleID/PID-ER_v2/models/'
plot_dir = '/lustrehome/adipilato/ParticleID/PID-ER_v2/plots/'
padding = 'padding' + str(height)
model_name= padding + '_ATmodel'
history_name = model_name + '_history'

class_labels = {22:0, -11:1, 11:1, -13:2, 13:2, -211:3, 211:3, 311:4, -1:5}
class_names = np.array(['γ', 'e$^{±}$', 'μ$^{±}$', 'π$^{±}$', 'K$^{0}$', 'inc'])


# In[3]:


# arrays of data needed for training

data_array = []
pid_array = []
en_array = []

# read dataset
files = [f for f in os.listdir(dataset_dir) if f.endswith("h5")]

for name in tqdm(files):
    print("Reading file", name)
    data = pd.read_hdf(dataset_dir + name)
    num_tracks = data.trackster.max()
    print(num_tracks)
    
    for i in range(1,num_tracks+1):
        track = data[data['trackster'] == i]
        img = np.array([track.E.values, track.eta.values, track.phi.values]).T.reshape(width, height, channels)
#         img = np.array([track.E.values, np.abs(track.eta.values-track.eta_mean.values), np.abs(track.phi.values-track.phi_mean.values)]).T.reshape(width, height, channels)
#         img = np.array([track.E.values, track.x_pca.values, track.y_pca.values, track.z_pca.values]).T.reshape(width, height, channels)
        pid_val = np.unique(track[track['pid'] != 0].pid)[0]
        pid = class_labels[int(pid_val)]
        en_value = track.genE.max()
        data_array.append(img)
        pid_array.append(pid)
        en_array.append(en_value)

data_array = np.array(data_array)
pid_array = np.array(pid_array)
pid_array_cat = keras.utils.to_categorical(pid_array, num_classes=classes, dtype='float32')
en_array = np.array(en_array)


# In[4]:


print(data_array.shape)
print(pid_array.shape)
print(en_array.shape)


# In[5]:


####### NORMALIZE THE ENERGY ########

# mean_en = 213.90352475881576
# std_en = 108.05413626100672
mean_en = np.mean(en_array)
std_en = np.std(en_array)

print('Mean Energy Value: {}'.format(mean_en))
print('Std Energy Value: {}'.format(std_en))

en_array_norm = (en_array - mean_en)/std_en


# In[6]:


# Load the trained model
model = load_model(save_dir + model_name + '.h5')
model.summary()


# In[7]:


# Score trained model

scores = model.evaluate(data_array, {'pid_output': pid_array_cat, 'enreg_output': en_array_norm}, verbose=1)
print("Scores: {}".format(scores))


# In[8]:


# Perform inference

results = model.predict(data_array)

print('****** PID START*******')
print('True Particle IDs= {} '.format(pid_array))
pid_results = results[0]
pid_predicted = np.argmax(pid_results, axis=1)
print('Predicted Particle ID Probabilities= {} '.format(pid_results))
print('Predicted Particle IDs= {} '.format(pid_predicted))
print('****** PID END*******')

print('****** ENREG START*******')
print('True Particle Energies= {} '.format(en_array))
enreg_results = results[1]
enreg_results = (enreg_results * std_en) + mean_en
enreg_results = np.squeeze(enreg_results)
print('Predicted Particle Energies= {}'.format(enreg_results))
print('****** ENREG END*******')


# In[9]:


gamma_true_en = en_array[pid_array==0]
electron_true_en = en_array[pid_array==1]
muon_true_en = en_array[pid_array==2]
pion_c_true_en = en_array[pid_array==3]
kaon_n_true_en = en_array[pid_array==4]

gamma_reco_en = enreg_results[pid_array==0]
electron_reco_en = enreg_results[pid_array==1]
muon_reco_en = enreg_results[pid_array==2]
pion_c_reco_en = enreg_results[pid_array==3]
kaon_n_reco_en = enreg_results[pid_array==4]


# In[10]:


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=13, y=1.04)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=12)

    plt.ylabel('True class', labelpad=10, fontsize=13)
    plt.xlabel('Predicted class', labelpad=10, fontsize=13)
    plt.tight_layout()


# In[11]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(pid_array, pid_predicted)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(0)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix')
plt.savefig(plot_dir + model_name + '_confusion_matrix.pdf', format='pdf')
plt.show()


# In[12]:


# #gamma's energy
# fig1 = plt.figure(1)
# plt.scatter(gamma_true_en, gamma_reco_en, color='red')
# plt.title('Gamma Energy Regression', y=1.04)
# plt.xlabel('True Energy', labelpad=8, fontsize=14)
# plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
# plt.plot([0,450],[0,450], color='black',linestyle='--')
# plt.grid(linestyle=':')
# plt.savefig(plot_dir + model_name + '_gammaEn.pdf', format='pdf')
# fig1.show()


# In[13]:


# #electron's energy
# fig2 = plt.figure(2)
# plt.scatter(electron_true_en, electron_reco_en, color='blue')
# plt.title('Electron Energy Regression', y=1.04)
# plt.xlabel('True Energy', labelpad=8, fontsize=14)
# plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
# plt.plot([0,450],[0,450], color='black',linestyle='--')
# plt.grid(linestyle=':')
# plt.savefig(plot_dir + model_name + '_electronEn.pdf', format='pdf')
# fig2.show()


# In[14]:


# #muon's energy
# fig3 = plt.figure(3)
# plt.scatter(muon_true_en, muon_reco_en, color='green')
# plt.title('Muon Energy Regression', y=1.04)
# plt.xlabel('True Energy', labelpad=8, fontsize=14)
# plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
# plt.plot([0,450],[0,450], color='black', linestyle='--')
# plt.grid(linestyle=':')
# plt.savefig(plot_dir + model_name + '_muonEn.pdf', format='pdf')
# fig3.show()


# In[15]:


# #pion_c's energy
# fig4 = plt.figure(4)
# plt.scatter(pion_c_true_en, pion_c_reco_en, color='violet')
# plt.title('Charged Pion Energy Regression', y=1.04)
# plt.xlabel('True Energy', labelpad=8, fontsize=14)
# plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
# plt.plot([0,450],[0,450], color='black', linestyle='--')
# plt.grid(linestyle=':')
# plt.savefig(plot_dir + model_name + '_pion_cEn.pdf', format='pdf')
# fig4.show()


# In[16]:


file = pd.read_hdf(save_dir + history_name + ".h5", "history") #.values
print(file.head())

file = file.values

val_loss = file[:, 0]
val_pid_loss = file[:, 1]
val_en_loss = file[:,2]

val_pid_acc =file[:,3]

train_loss = file[:, 5]
train_pid_loss = file[:,6]
train_en_loss = file[:,7]

train_pid_acc = file[:, 8]

n_epochs = len(file)
n_epochs = np.arange(1, n_epochs+1)
print("Number of Epochs: ", n_epochs)


# In[17]:


fig5 = plt.figure(5)
plt.plot(n_epochs, train_pid_acc, '-b', label='Training')
plt.plot(n_epochs, val_pid_acc, '-r', label='Validation')

plt.title('Model PID accuracy', y=1.04)
plt.grid(linestyle=':')
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Accuracy', labelpad=10, fontsize=14)
plt.legend(loc='lower right')
plt.savefig(plot_dir + model_name + '_pid_accuracy.pdf', format='pdf')
fig5.show()


# In[18]:


fig6 = plt.figure(6)
plt.plot(n_epochs, train_loss, '-b', label='Training')
plt.plot(n_epochs, val_loss, '-r', label='Validation')

plt.title('Model loss function', y=1.04)
plt.grid(linestyle=':')
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Loss', labelpad=10, fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(plot_dir + model_name + '_total_loss.pdf', format='pdf')
fig6.show()


# In[19]:


fig7 = plt.figure(7)
plt.plot(n_epochs, train_loss, '-g', label='Total')
plt.plot(n_epochs, train_pid_loss, '-b', label='PID')
plt.plot(n_epochs, train_en_loss, '-r', label='ER')
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 13)
ax.tick_params(axis = 'y', which = 'major', labelsize = 13)
plt.title('Model training loss function', y=1.04, fontsize=14)
plt.grid(linestyle=':')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Loss', labelpad=8, fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(plot_dir + model_name + '_training_loss.pdf', format='pdf')
fig7.show()


# In[20]:


fig8 = plt.figure(8)
plt.plot(n_epochs, val_loss, '-g', label='Total')
plt.plot(n_epochs, val_pid_loss, '-b', label='PID')
plt.plot(n_epochs, val_en_loss, '-r', label='ER')
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 13)
ax.tick_params(axis = 'y', which = 'major', labelsize = 13)
plt.title('Model validation loss function', y=1.04, fontsize=14)
plt.grid(linestyle=':')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Loss', labelpad=8, fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(plot_dir + model_name + '_validation_loss.pdf', format='pdf')
fig8.show()


# In[21]:


#gamma's energy hist
fig9 = plt.figure(9)
plt.hist2d(gamma_true_en, gamma_reco_en, (50,50), cmap='hot_r', range=[[0, 450], [0, 450]])
plt.title('γ energy regression', y=1.04, fontsize=14)
plt.xlabel('True Energy (GeV)', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy (GeV)', labelpad=8, fontsize=14)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 13)
ax.tick_params(axis = 'y', which = 'major', labelsize = 13)
plt.plot([0,450],[0,450], color='red',linestyle='-')
# plt.grid(linestyle=':')
cbar = plt.colorbar()
cbar.set_label('Counts', fontsize=14, labelpad=8)
plt.tight_layout()
plt.savefig(plot_dir + model_name + '_gammaEnHist.pdf', format='pdf')
fig9.show()


# In[22]:


#electron's energy hist
fig10 = plt.figure(10)
plt.hist2d(electron_true_en, electron_reco_en, (50,50), cmap='hot_r', range=[[0, 450], [0, 450]])
plt.title('e$^{-}$ energy regression', y=1.04, fontsize=14)
plt.xlabel('True Energy (GeV)', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy (GeV)', labelpad=8, fontsize=14)
plt.xlim(0,450)
plt.ylim(0,450)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 13)
ax.tick_params(axis = 'y', which = 'major', labelsize = 13)
plt.plot([0,450],[0,450], color='red',linestyle='-')
# plt.grid(linestyle=':')
cbar = plt.colorbar()
cbar.set_label('Counts', fontsize=14, labelpad=8)
plt.tight_layout()
plt.savefig(plot_dir + model_name + '_electronEnHist.pdf', format='pdf')
fig10.show()


# In[23]:


#pion's energy hist
fig11 = plt.figure(11)
plt.hist2d(pion_c_true_en, pion_c_reco_en, (50,50), cmap='hot_r', range=[[0, 450], [0, 450]])
plt.title('π$^{+}$ energy regression', y=1.04, fontsize=14)
plt.xlabel('True Energy (GeV)', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy (GeV)', labelpad=8, fontsize=14)
plt.xlim(0,450)
plt.ylim(0,450)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 13)
ax.tick_params(axis = 'y', which = 'major', labelsize = 13)
plt.plot([0,450],[0,450], color='red',linestyle='-')
# plt.grid(linestyle=':')
cbar = plt.colorbar()
cbar.set_label('Counts', fontsize=14, labelpad=8)
plt.tight_layout()
plt.savefig(plot_dir + model_name + '_pion_cEnHist.pdf', format='pdf')
fig11.show()


# In[ ]:


#kaon's energy hist
fig12 = plt.figure(12)
plt.hist2d(kaon_n_true_en, kaon_n_reco_en, (50,50), cmap='hot_r', range=[[0, 450], [0, 450]])
plt.title('K$^{0}$ energy regression', y=1.04, fontsize=14)
plt.xlabel('True Energy (GeV)', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy (GeV)', labelpad=8, fontsize=14)
plt.xlim(0,450)
plt.ylim(0,450)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 13)
ax.tick_params(axis = 'y', which = 'major', labelsize = 13)
plt.plot([0,450],[0,450], color='red',linestyle='-')
# plt.grid(linestyle=':')
cbar = plt.colorbar()
cbar.set_label('Counts', fontsize=14, labelpad=8)
plt.tight_layout()
plt.savefig(plot_dir + model_name + '_kaon_nEnHist.pdf', format='pdf')
fig11.show()

