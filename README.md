# PID-ER_v2
Repository for Particle ID and Energy Regression on vinavx2 @ CERN


Root files can be found @ /afs/cern.ch/user/a/adipilat/public . They're obtained with NTUPLIZER, after running step3 with TICL and saving Tracksters.

The model is a CNN that splits in two branches at the end and outputs PID and ER for 4 classes (electron, gamma, muon, charged pion).

The .h5 generated have no padding or cuts, as they're performed in the training/inference file. This allows us to customize more the way we perform the training. The four .h5 are not merged, since a single root file will be created when shooting a random number of particles in the next step of this study (thus it's not necessary). Clusters are already ordered by decreasing energy values, making easier to perform cuts or padding before training.

