# IndNet
Defining subject-specific brain networks by individualizing templates

## Introduction
IndNet is a Nipype inplementation of a dual-regression-like seed-based approach to individualize general binary templates to specific subjects. Templates are assumed to be in MNI space.
Raw resting-state and anatomical images are preprocessed and transformed into MNI space. Timecourses are extracted from the resting state images for all templates (using the first eigenvariate) and fed into a GLM analysis, together with mean WM and CSF signals. Specific contrasts, representing networks of interest, are tested and results are thresholded using spatial mixture modeling and subsequently binarized. In addition to the resulting main network maps (which might be overlapping), IndNet also outputs exclusive maps of the networks of interest (i.e. non-overlapping maps). All maps are in MNI space.

<a href="https://github.com/can-lab/IndNet/blob/master/indnet_graph_simple.png">
  <img src="https://github.com/can-lab/IndNet/raw/master/indnet_graph_simple.png" width="500">
</a>


## Prerequisites
1. Install [Nipype](https://nipype.readthedocs.io)
2. Install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
3. Install [graphviz](https://www.graphviz.org/)
4. Install ICA-AROMA
   - [Download](https://github.com/fladd/ICA-AROMA/archive/master.zip)
   - Install dependencies (see `requirements.txt`)
   - Make `ICA_AROMA.py` executable
   - Add `ICA_AROMA.py` to path, so it can be called system wide
5. [Download IndNet](https://github.com/can-lab/IndNet/archive/master.zip)

### Donders cluster
If you are working on the compute cluster of the Donders Institute, please follow the following steps:
1. Load Python module by running command: `module load python`
2. Install virtualenv by running command: `python -m pip install virtualenv --user`
3. Create new environment in home directory by running command: `cd && python -m virtualenv IndNet`
4. Activate new environment by running command: `source ~/IndNet/bin/activate`
5. Update pip in environment by running command: `pip install -U pip`
6. Install Nipype into environment by running command: `pip install nipype`
7. Install ICA-AROMA
   - [Download](https://github.com/fladd/ICA-AROMA/archive/master.zip) and extract `ICA-Aroma-master` to somewhere
   - Within `ICA-Aroma-master`
      - Install dependencies by running command: `pip install -r requirements.txt`
      - Make `ICA_AROMA.py` executable by running command: `chmod a+x ICA_AROMA.py`
   - Add `ICA_AROMA.py` to path, so it can be called system wide
      - Open file `~/.bash_profile` in a text editor and add: `export PATH=/path/to/ICA-Aroma-master/:$PATH`
      - Run command `source ~/.bash_profile`
8. [Download IndNet](https://github.com/can-lab/IndNet/archive/master.zip) and extract `IndNet-master` to somewhere

## Usage
1. Write script with custom workflow (see `Indnet-master/example.py` for an example)
2. Run script
3. Results are in `results` directory within the `base_dir` set in the script

### Donders cluster
If you are working on the compute cluster of the Donders Institute, please follow the following steps:
1. Start a new interactive job by running command: `qsub -I -l 'procs=8, mem=16gb, walltime=4:00:00'`
2. Load Python module by running command: `module load python`
3. Load graphviz module by running command: `module load graphviz`
4. Activate environment by running command: `source ~/IndNet/bin/activate`
5. Write script with custom workflow (see `Indnet-master/example.py` for an example) and save it as `my_script.py`
6. Run script by running command: `python my_script.py`
7. Results are in `results` directory within the `base_dir` set in the script

When done, deactivate environment by running command: `source ~/IndNet/bin/deactivate`


