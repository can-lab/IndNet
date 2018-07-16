"""Example IndNet analysis.

Use IndNet to find subject-specific SN, ECN and DMN, based on a combination of
network templates from the FIND lab at Stanford University (which can be
downloaded from http://findlab.stanford.edu/functional_ROIs.html).

"""

__author__ = "Florian Krause"


import os
from glob import glob

from nipype.interfaces import io
from nipype.pipeline.engine import Node, Workflow

from indnet import create_indnet_workflow


# Set up a workflow
core_networks = Workflow(name='core_networks')
core_networks.base_dir = "/path/to/base_directory/"  # set working/output directory

# Create indnet node
indnet = create_indnet_workflow(smoothing=5, hp_cutoff=100, threshold=0.66, aggr_aroma=False)
indnet.inputs.inputspec.t1_file = "/path/to/t1.nii"  # point to anatomical T1 scan (NiFTI file)
indnet.inputs.inputspec.rs_file = "/path/to/rs.nii"  # point to functional resting state scan (NiFTI file)
TEMPL_DIR = os.path.abspath("./Functional_ROIs")  # point to FIND template directory
indnet.inputs.inputspec.templates = [
    os.path.join(TEMPL_DIR, 'anterior_Salience', "anterior_Salience.nii.gz"),
    os.path.join(TEMPL_DIR, 'Auditory', 'Auditory.nii.gz'),
    os.path.join(TEMPL_DIR, 'Basal_Ganglia', 'Basal_Ganglia.nii.gz'),
    os.path.join(TEMPL_DIR, 'dorsal_DMN', 'dDMN.nii.gz'),
    os.path.join(TEMPL_DIR, 'high_Visual', 'high_Visual.nii.gz'),
    os.path.join(TEMPL_DIR, 'Language', 'Language.nii.gz'),
    os.path.join(TEMPL_DIR, 'LECN', 'LECN.nii.gz'),
    os.path.join(TEMPL_DIR, 'post_Salience', 'post_Salience.nii.gz'),
    os.path.join(TEMPL_DIR, 'Precuneus', 'Precuneus.nii.gz'),
    os.path.join(TEMPL_DIR, 'prim_Visual', 'prim_Visual.nii.gz'),
    os.path.join(TEMPL_DIR, 'RECN', 'RECN.nii.gz'),
    os.path.join(TEMPL_DIR, 'Sensorimotor', 'Sensorimotor.nii.gz'),
    os.path.join(TEMPL_DIR, 'ventral_DMN', 'vDMN.nii.gz'),
    os.path.join(TEMPL_DIR, 'Visuospatial', 'Visuospatial.nii.gz'),
]
indnet.inputs.inputspec.networks = [
    {'name': 'SN', 'components': [0, 7]},
    {'name': 'ECN', 'components': [6, 10]},
    {'name': 'DMN', 'components': [3, 12]},
]

# Create results node
results = Node(io.DataSink(parameterization=False), name='results')

# Connect indnet and output nodes
core_networks.connect(indnet, 'outputspec.mainfiles',
                      results, 'main')
core_networks.connect(indnet, 'outputspec.exclusivefiles',
                      results, 'exclusive')


# Run workflow
core_networks.write_graph(dotfilename='core_networks_graph.dot', graph2use='colored')
core_networks.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
