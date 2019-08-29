import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("Demo")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D41Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
try:
    process.load('RecoLocalCalo.HGCalRecProducers.HGCalLocalRecoSequence_cff')
except Exception: # ConfigFileReadError in case config does not exist
    process.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi')
    process.load('RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
from FastSimulation.Event.ParticleFilter_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:step3.root'
    ),
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck")
)


process.ana = cms.EDAnalyzer('HGCalTracksterPID',
                             detector = cms.string("all"),
                             inputTag_HGCalMultiCluster = cms.string("hgcalMultiClusters"),
                             rawRecHits = cms.bool(False),
                             readCaloParticles = cms.bool(True),
                             storeGunParticles = cms.bool(True),
			                 storeGenParticleOrigin = cms.bool(True),
                             storeGenParticleExtrapolation = cms.bool(True),
                             storePCAvariables = cms.bool(False),
                             storeElectrons = cms.bool(False),
                             storePFCandidates = cms.bool(False),
                             readGenParticles = cms.bool(True),
                             recomputePCA = cms.bool(False),
                             includeHaloPCA = cms.bool(False),
                             dEdXWeights = dEdX.weights,
                             layerClusterPtThreshold = cms.double(-1),  # All LayerCluster belonging to a multicluster are saved; this Pt threshold applied to the others
                             TestParticleFilter = ParticleFilterBlock.ParticleFilter,
                             doTracksters = cms.bool(True),
                             storeCPMatching = cms.bool(True),
                             forPIDOnly = cms.bool(True),
		                     verbose=cms.bool(True)
)

process.ana.TestParticleFilter.protonEMin = cms.double(100000)
process.ana.TestParticleFilter.etaMax = cms.double(3.1)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("file:step4.root")

                                   )

reRunClustering = False

if reRunClustering:
    #process.hgcalLayerClusters.minClusters = cms.uint32(0)
    #process.hgcalLayerClusters.realSpaceCone = cms.bool(True)
    #process.hgcalLayerClusters.multiclusterRadius = cms.double(2.)  # in cm if realSpaceCone is true
    #process.hgcalLayerClusters.dependSensor = cms.bool(True)
    #process.hgcalLayerClusters.ecut = cms.double(3.)  # multiple of sigma noise if dependSensor is true
    #process.hgcalLayerClusters.kappa = cms.double(9.)  # multiple of sigma noise if dependSensor is true
    #process.hgcalLayerClusters.deltac = cms.vdouble(2.,3.,5.) #specify delta c for each subdetector separately
    process.p = cms.Path(process.hgcalLayerClusters+process.ana)
else:
    process.p = cms.Path(process.ana)
