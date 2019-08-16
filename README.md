# PID-ER_v2
Repository for Particle ID and Energy Regression on vinavx2 @ CERN


Root files can be found @ /afs/cern.ch/user/a/adipilat/public . They're obtained with NTUPLIZER, after running step3 with TICL and saving Tracksters. An example of step3(withTICL) and step4(NTUPLIZER) are included in the repository. To set everything for production, refer to this branch of my cmssw fork: https://github.com/tonydp03/cmssw/tree/PIDandER_TICL .

The model is a CNN that splits in two branches at the end and outputs PID and ER for 4 classes (electron, gamma, muon, charged pion).

The .h5 generated have no padding or cuts, as they're performed in the training/inference file. This allows us to customize more the way we perform the training. The four .h5 are not merged, since a single root file will be created when shooting a random number of particles in the next step of this study (thus it's not necessary). Clusters are already ordered by decreasing energy values, making easier to perform cuts or padding before training.

Since we are shooting a single particle, only the leading Trackster is considered for training (the most energetic one) with the generated energy value. The code is written such that when we'll shoot more particles, we'll save all the tracksters thus generated (there's an 'if' condition that will be removed at that point). In that case, the association caloparticle -> trackster will be added and the energy of the caloparticle used instead of the generated energy value.

The only issue now is that the pre-processing phase (reading data into lists, padding, conversion to np.array) is slow (it's being done for the leading trackster only but it takes around 10 minutes for 10k events). Padding the dataframe would have been more complicated since all tracksters are saved and the number of tracksters per event is not constant.