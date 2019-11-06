import sys
import uproot
import os
import numpy as np
import pandas as pd

import argparse

path = "/lustre/cms/store/user/adiflori/HGCal_PID_pGuns/new_data/"
unpad_path = "/lustrehome/adipilato/ParticleID/new_datasets/5PartPerEvent/unpadded/"
pad_path = "/lustrehome/adipilato/ParticleID/new_datasets/5PartPerEvent/padded/"
dir_ = "ana"
tree = "hgc"
max_perlayer = 10
number_layers = 50

variableName = [
            'event',
            'cluster2d_layer',
            'cluster2d_energy',
            'cluster2d_eta',
            'cluster2d_phi',
            'cluster2d_pt',
            'cluster2d_x',
            'cluster2d_y',
            'cluster2d_z',
            'gen_pdgid',
            'gen_energy',
            'cluster2d_best_cpPdg',
            'cluster2d_best_cpId',
            'tracksterEM_clusters',
            'tracksterMIP_clusters',
            'tracksterHAD_clusters'
            ]
newVars =["event","tracksterID","trackster","layer","x","y","z","phi","eta","E","pt","genE","pid"]

parser = argparse.ArgumentParser()
parser.add_argument('--num',  type=int, default="0")
args = parser.parse_args()


filenum = args.num
name = "step4_" + str(filenum)
filename = path + name + ".root"
print("Starting data production for " + filename)


# start_event = 0

# Here goes the loop
df = uproot.open(filename)[dir_][tree].pandas.df(variableName,flatten=False)

xs = df["cluster2d_x"].values
ys = df["cluster2d_y"].values
zs = df["cluster2d_z"].values
es = df["cluster2d_energy"].values
ps = df["cluster2d_pt"].values
ll = df["cluster2d_layer"].values
cphi = df["cluster2d_phi"].values
ceta = df["cluster2d_eta"].values
cp = df["cluster2d_best_cpPdg"].values
cpid = df["cluster2d_best_cpId"].values
    
sizes = [x.shape[0] for x in xs]
ee = np.arange(1, len(sizes)+1) # + start_event
num_events = len(sizes)

gen = df["gen_energy"].values
genpdg = df["gen_pdgid"].values

trEM = df["tracksterEM_clusters"].values
trMIP = df["tracksterMIP_clusters"].values
trHAD = df["tracksterHAD_clusters"].values

# Define some trackster type labels - 0 = EM, 1 = HAD, 2 = MIP --->not used now
typeTr = [trEM,trHAD] #,trMIP]

# Store all the CP recoEn per event
cprecoEn = [] 
for i in range(num_events):
    tempEn = []
    cps = np.unique(cpid[i])
    for j in range(len(cps)):
        indices = np.where(cpid[i]==cps[j])
        tempEn.append(sum(es[i][indices]))
    cprecoEn.append(tempEn)


# Declare new lists

evId,trId,trNum,xTr,yTr,zTr,lTr,ptTr,enTr,etaTr,phiTr,pidTr,genTr = [],[],[],[],[],[],[],[],[],[],[],[],[]

# Loop over tracksters and append the LC info to the respective lists

for i in range(num_events):
    if(len(np.unique(cpid[i])) < 5):
        continue
    for num, key in enumerate(typeTr):
        for j in range(len(key[i])):
            cpidTr = []
            for item in key[i][j]:
                evId.append(ee[i])
                trId.append(num)
                trNum.append(j)
                xTr.append(xs[i][item])
                yTr.append(ys[i][item])
                zTr.append(zs[i][item])
                lTr.append(ll[i][item])
                ptTr.append(ps[i][item])
                enTr.append(es[i][item])
                etaTr.append(ceta[i][item])
                phiTr.append(cphi[i][item])

                cpidTr.append(cpid[i][item])
            cpIdx = np.unique(cpidTr)
            cpIdx = cpIdx[cpIdx != 4294967295] # should be -1    
            fracEn = []

            for k in (cpIdx):
                indices = np.where(cpidTr==k)
                track_idx = [key[i][j][l] for l in indices[0]]
                frac = float(sum(es[i][track_idx])/cprecoEn[i][k])
                fracEn.append(frac)

            maxfracIdx = np.argmax(fracEn)
            caloIdx = cpIdx[maxfracIdx]
            if(fracEn[maxfracIdx] > 0.5):
                indices = np.where(cpidTr==cpIdx[maxfracIdx])
                track_idx = [key[i][j][l] for l in indices[0]]
                pidTr.append([cp[i][track_idx[0]]]*len(key[i][j]))
                genTr.append([gen[i][caloIdx]]*len(key[i][j]))
            else:
                pidTr.append([-1]*len(key[i][j]))
                genTr.append([0]*len(key[i][j]))


# Flatten everything and create arrays for dataset
EVID = np.array(evId)
TRID = np.array(trId)
TRNUM = np.array(trNum)
XTR = np.array(xTr)
YTR = np.array(yTr)
ZTR = np.array(zTr)
LTR = np.array(lTr)
PTTR = np.array(ptTr)
ENTR = np.array(enTr)
ETATR = np.array(etaTr)
PHITR = np.array(phiTr)
GENTR = np.array([item for sublist in genTr for item in sublist])
PIDTR = np.array([item for sublist in pidTr for item in sublist])


# Create the dataset
datas = np.vstack((EVID,TRID,TRNUM,LTR,XTR,YTR,ZTR,PHITR,ETATR,ENTR,PTTR,GENTR,PIDTR)).T
df = pd.DataFrame(datas,columns=newVars)
df = df.sort_values(["event","tracksterID","trackster","layer","E"],ascending=[True,True,True,True,False]).reset_index(drop=True)


df.to_hdf(unpad_path + name + "_unpadded.h5","data",complevel=0)

print('Unpadded dataset created!')

# Now we enumerate tracksters since they'll be used for training and we don't need the info
#about the event or the trackster type
trackster_sizes = df.groupby(['event', 'tracksterID', 'trackster']).size().values.tolist()
trackster_places = np.cumsum(trackster_sizes)
num_tracksters = len(trackster_sizes)
track_startes = np.array( [0] + list(trackster_places[:-1]))
track_finishes = np.array(list(track_startes[1:]) +[len(df)])
track_id = np.arange(1,num_tracksters+1)
track_bounds = np.vstack((track_startes,track_finishes)).T
new_tracks = [[i for j in range(t[1]-t[0])] for i,t in zip(track_id, track_bounds)]
new_tracks = np.array([item for sublist in new_tracks for item in sublist])
df['trackster'] = new_tracks


del df['event']
del df['tracksterID']

theIndex = list(df.groupby(["trackster","layer"]).indices.values())
theIndex = np.array([item for sublist in theIndex for item in sublist[:min(len(sublist),10)]])
df = df.iloc[theIndex]

# Introduce proper indices to copy the old dataset into the padded one
layer_sizes = df.groupby(["trackster","layer"]).size().values.tolist()
layer_places = np.cumsum(layer_sizes)

startes = np.array( [0] + list(layer_places[:-1]))
layers = df["layer"].values[startes]
ids = df["trackster"].values[startes]
finishes = np.array(list(startes[1:]) +[len(df)])
SSS = np.vstack((startes,finishes)).T

hitIds = [[j +(n-1)*max_perlayer + max_perlayer*number_layers*(e-1) for j in range(s[1]-s[0])] for n,s,e in zip(layers,SSS,ids)]
hitIds = np.array([item for sublist in hitIds for item in sublist])

df.loc[:,"hitIds"] = hitIds
df = df.set_index(hitIds.astype(int))

#Create the big mask and copy the old dataset in it to have to padded one
num_tracksters = df.trackster.max()    

bigMask = np.zeros((num_tracksters*number_layers*max_perlayer,len(df.columns)))
bigDF = pd.DataFrame(bigMask,columns=df.columns)

fakeHit = [ [(i*max_perlayer + j) for j in range(max_perlayer)] for i in range(number_layers*num_tracksters)]
fakeHit = np.array([item for sublist in fakeHit for item in sublist])

fakeLayer = [ np.full(max_perlayer,i) for j in range(1,num_tracksters+1) for i in range(1,number_layers+1)]
fakeLayer = np.array([item for sublist in fakeLayer for item in sublist])    

fakeTrackster = [ np.full(max_perlayer*number_layers,i) for i in range(1,num_tracksters+1)]
fakeTrackster = np.array([item for sublist in fakeTrackster for item in sublist])  

bigDF["layer"] = fakeLayer
bigDF["trackster"] = fakeTrackster
bigDF["hitIds"] = fakeHit

bigDF.iloc[df.index] = df
del bigDF['hitIds']

bigDF.to_hdf(pad_path + name + "_padded.h5","data",complevel=0)

print('Padded dataset created!')
