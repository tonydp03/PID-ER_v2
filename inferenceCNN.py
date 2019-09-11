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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from keras import backend as K

from sklearn.metrics import confusion_matrix
import itertools

from tqdm import tqdm

from PIL import Image

width = 50
height = 10
channels = 3
classes = 4
dataset_dir = '/data/user/adipilat/ParticleID/genEvts/'
save_dir = '/data/user/adipilat/ParticleID/models/'
plot_dir = '/data/user/adipilat/ParticleID/plots/'
padding = 'padding' + str(height)
model_name= padding +'_model'
history_name = model_name + '_history'


# This dictionary should be extended to new classes and antiparticles
class_labels = {22:0, 11:1, 13:2, 211:3}

class_names = np.array(['gamma', 'electron', 'muon', 'pion_c'])

# arrays of data needed for training
data_array = []
pid_array = []
en_array = []

samples = []

# read dataset
files = [f for f in os.listdir(dataset_dir) if f.endswith("h5")]

for name in tqdm(files):
    print("Reading file", name)
    data = pd.read_hdf(dataset_dir + name)

    # using 10% of events for test
    n_events_start = int(0.9 * data.event.max()) 
    n_events = int(data.event.max())
    
    for i in range(n_events_start+1, n_events+1):
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
        if(i == n_events_start+1):
            img = image[:,:,0]
            img2 = np.transpose(img, (1, 0))
            plt.imsave(plot_dir + 'sample_' + name + '.pdf', img, format='pdf')

    print("File", name, " processed")

data_array = np.array(data_array)
pid_array = np.array(pid_array)
pid_array_cat = keras.utils.to_categorical(pid_array, num_classes=classes, dtype='float32')
en_array = np.array(en_array)


print(data_array.shape)
print(en_array.shape)
print(pid_array.shape)


####### NORMALIZE THE ENERGY ########
# mean_en = np.mean(en_array)
# std_en = np.std(en_array)
mean_en = 213.90352475881576
std_en = 108.05413626100672

print('Mean Energy Value: {}'.format(mean_en))
print('Std Energy Value: {}'.format(std_en))

en_array_norm = (en_array - mean_en)/std_en

# Load the trained model
model = load_model(save_dir + model_name + '.h5')

# Score trained model
scores = model.evaluate(data_array, {'pid_output': pid_array_cat, 'enreg_output': en_array_norm}, verbose=1)
print("Scores: {}".format(scores))


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

# Prepare data for ER plots
gamma_true_en = en_array[pid_array==0]
electron_true_en = en_array[pid_array==1]
muon_true_en = en_array[pid_array==2]
pion_c_true_en = en_array[pid_array==3]

gamma_reco_en = enreg_results[pid_array==0]
electron_reco_en = enreg_results[pid_array==1]
muon_reco_en = enreg_results[pid_array==2]
pion_c_reco_en = enreg_results[pid_array==3]


def plot_confusion_matrix(cm, classes, normalize=True, title='Normalized confusion matrix', cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=7)
    plt.yticks(tick_marks, classes, fontsize=7)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)

    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(pid_array, pid_predicted)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
fig0 = plt.figure(0)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.savefig(plot_dir + model_name + '_confusion_matrix.pdf', format='pdf')
#fig0.show()


#gamma's energy
fig1 = plt.figure(1)
plt.scatter(gamma_true_en, gamma_reco_en, color='red')
plt.title('Gamma Energy Regression', y=1.04)
plt.xlabel('True Energy', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
plt.plot([0,450],[0,450], color='black')
plt.grid(linestyle=':')
plt.savefig(plot_dir + model_name + '_gammaEn.pdf', format='pdf')
#fig1.show()

#electron's energy
fig2 = plt.figure(2)
plt.scatter(electron_true_en, electron_reco_en, color='blue')
plt.title('Electron Energy Regression', y=1.04)
plt.xlabel('True Energy', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
plt.plot([0,450],[0,450], color='black')
plt.grid(linestyle=':')
plt.savefig(plot_dir + model_name + '_electronEn.pdf', format='pdf')
#fig2.show()

#muon's energy
fig3 = plt.figure(3)
plt.scatter(muon_true_en, muon_reco_en, color='green')
plt.title('Muon Energy Regression', y=1.04)
plt.xlabel('True Energy', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
plt.plot([0,450],[0,450], color='black')
plt.grid(linestyle=':')
plt.savefig(plot_dir + model_name + '_muonEn.pdf', format='pdf')
#fig3.show()

#pion_c's energy
fig4 = plt.figure(4)
plt.scatter(pion_c_true_en, pion_c_reco_en, color='violet')
plt.title('Charged Pion Energy Regression', y=1.04)
plt.xlabel('True Energy', labelpad=8, fontsize=14)
plt.ylabel('Predicted Energy', labelpad=10, fontsize=14)
plt.plot([0,450],[0,450], color='black')
plt.grid(linestyle=':')
plt.savefig(plot_dir + model_name + '_pion_cEn.pdf', format='pdf')
#fig4.show()

file = pd.read_hdf(save_dir + history_name + ".h5", "history") 
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


fig5 = plt.figure(5)
plt.plot(n_epochs, train_pid_acc, '-b', label='Training')
plt.plot(n_epochs, val_pid_acc, '-r', label='Validation')
plt.title('Model PID accuracy', y=1.04)
plt.grid(linestyle=':')
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Accuracy', labelpad=10, fontsize=14)
plt.legend(loc='lower right')
plt.savefig(plot_dir + model_name + '_pid_accuracy.pdf', format='pdf')
#fig5.show()

fig6 = plt.figure(6)
plt.plot(n_epochs, train_loss, '-b', label='Training')
plt.plot(n_epochs, val_loss, '-r', label='Validation')
plt.title('Model loss function', y=1.04)
plt.grid(linestyle=':')
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Loss', labelpad=10, fontsize=14)
plt.legend(loc='upper right')
plt.savefig(plot_dir + model_name + '_total_loss.pdf', format='pdf')
#fig6.show()

fig7 = plt.figure(7)
plt.plot(n_epochs, train_loss, '-g', label='Total')
plt.plot(n_epochs, train_pid_loss, '-b', label='PID')
plt.plot(n_epochs, train_en_loss, '-r', label='ER')
plt.title('Model training loss functions', y=1.04)
plt.grid(linestyle=':')
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Loss', labelpad=10, fontsize=14)
plt.legend(loc='upper right')
plt.savefig(plot_dir + model_name + '_training_loss.pdf', format='pdf')
#fig7.show()

fig8 = plt.figure(8)
plt.plot(n_epochs, val_loss, '-g', label='Total')
plt.plot(n_epochs, val_pid_loss, '-b', label='PID')
plt.plot(n_epochs, val_en_loss, '-r', label='ER')
plt.title('Model validation loss functions', y=1.04)
plt.grid(linestyle=':')
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Loss', labelpad=10, fontsize=14)
plt.legend(loc='upper right')
plt.savefig(plot_dir + model_name + '_validation_loss.pdf', format='pdf')
#fig8.show()
