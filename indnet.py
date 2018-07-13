import os
import subprocess
from glob import glob

import numpy as np
import scipy.stats as stats

from nipype import Node, MapNode, Workflow
from nipype.interfaces import utility, fsl
from nipype.interfaces.base import BaseInterface, \
        BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename
from nipype.workflows.fmri.fsl import create_featreg_preproc, \
        create_reg_workflow


class ZtransformInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='textfile with values to be z-transformed',
                   mandatory=True)


class ZtransformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="textfile with z-transformed values")


class Ztransform(BaseInterface):
    input_spec = ZtransformInputSpec
    output_spec = ZtransformOutputSpec

    def _run_interface(self, runtime):
        fname = self.inputs.in_file
        with open(fname) as f:
            data = [float(x.strip()) for x in f.readlines()]
        zdata = stats.mstats.zscore(np.array(data), axis=0)
        _, base, ext = split_filename(fname)
        with open(base + '_ztransformed' + ext, 'w') as f:
            for x in zdata:
                f.write(str(x) + "\n")
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        _, base, ext = split_filename(fname)
        outputs["out_file"] = os.path.abspath(base + '_ztransformed' + ext)
        return outputs


class DesignMatrixInputSpec(BaseInterfaceInputSpec):
    in_files = traits.List(
        traits.Any, mandatory=True, desc='list of one-column regressors textfiles')


class DesignMatrixOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="design file to be used in GLM")


class DesignMatrix(BaseInterface):
    input_spec = DesignMatrixInputSpec
    output_spec = DesignMatrixOutputSpec

    def _run_interface(self, runtime):
        data = []
        for fname in self.inputs.in_files:
            with open(fname) as f:
                data.append([x.strip() for x in f.readlines()])
        with open("design.txt", 'w') as f:
            for line in range(len(data[0])):
                f.write(" ".join([column[line] for column in data]) + "\n")
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.abspath('design.txt')
        return outputs


class ContrastsInputSpec(BaseInterfaceInputSpec):
    in_list = traits.List(
        traits.Any, mandatory=True, desc='list of contrasts')
    design = File(exists=True, desc="design file to be used in GLM")


class ContrastsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="contrast file to be used in GLM")


class Contrasts(BaseInterface):
    input_spec = ContrastsInputSpec
    output_spec = ContrastsOutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.design) as f:
            lines = [x.strip() for x in f.readlines()]
            n = len(lines[0].split(" "))

        # Create main effect
        contrasts = []
        for x in self.inputs.in_list:
            contrast = [0] * n
            for y in x:
                contrast[y] = 1
            contrasts.append(contrast)

        # Create all differences to networks of interest
        contrasts2 = []
        for x in contrasts:
            current = [i for i, j in enumerate(x) if j == 1]
            for y in contrasts:
                if y != x:
                    other = [i for i, j in enumerate(y) if j == 1]
                    contrast = x[:]
                    for z in current:
                        contrast[z] = (len(current) * len(other)) \
                                / float(len(current))
                    for z in other:
                        contrast[z] = -(len(current) * len(other)) \
                                / float(len(other))
                    contrasts2.append(contrast)
            
        with open('contrasts.txt', 'w') as f:
            contrasts.extend(contrasts2)
            for contrast in contrasts:
                f.write(" ".join([str(x) for x in  contrast]) + "\n")
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.abspath('contrasts.txt')
        return outputs


class SplitMapsInputSpec(BaseInterfaceInputSpec):
    in_files = traits.List(
        traits.Any, mandatory=True, desc='list of binary maps')
    in_networks = traits.List(
        traits.Any, mandatory=True, desc='list of networks')


class SplitMapsOutputSpec(TraitedSpec):
    out_mains = traits.List(
        traits.Any, mandatory=True, desc='list of binary maps')
    out_firsts = traits.List(
        traits.Any, mandatory=True, desc='list of binary maps')
    out_opstrings = traits.List(
        traits.Any, mandatory=True, desc='list of opstrings for fslmaths')


class SplitMaps(BaseInterface):
    input_spec = SplitMapsInputSpec
    output_spec = SplitMapsOutputSpec

    def _run_interface(self, runtime):
        n = len(self.inputs.in_networks)
        self.out_mains = self.inputs.in_files[:n]
        self.out_firsts = []
        self.out_opstrings = []
        for m in range(n, len(self.inputs.in_files), n - 1):
            self.out_firsts.append(self.inputs.in_files[m])
            l = []
            for elem in range(m + 1, m + n - 1):
                l.append(self.inputs.in_files[elem])
            if l != []:
                self.out_opstrings.append('-mul ' + ' -mul'.join(l)) 
        
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_mains"] = self.out_mains
        outputs["out_firsts"] = self.out_firsts
        outputs["out_opstrings"] = self.out_opstrings
        return outputs


def create_highpass_filter(cutoff=100, name='highpass'):
    highpass = Workflow(name=name)
    inputspec = Node(utility.IdentityInterface(fields=['in_file']),
                     name='inputspec')

    # calculate sigma
    def calculate_sigma(in_file, cutoff):
        import subprocess
        output = subprocess.check_output(['fslinfo', in_file]).split("\n")
        for out in output:
            if out.startswith("pixdim4"):
                sigma = cutoff / (2 * float(out.lstrip("pixdim4")))
                return '-bptf %.10f -1' % sigma

    getsigma = Node(utility.Function(function=calculate_sigma,
                                     input_names=['in_file', 'cutoff'],
                                     output_names=['op_string']),
                                     name='getsigma')
    getsigma.inputs.cutoff = cutoff

    # save mean
    meanfunc = Node(fsl.ImageMaths(op_string='-Tmean', suffix='_mean'),
                    name='meanfunc')

    # filter data
    filter_ = Node(fsl.ImageMaths(suffix='_tempfilt'), name='filter')

    # restore mean
    addmean = Node(fsl.BinaryMaths(operation='add'), name='addmean')

    outputspec = Node(utility.IdentityInterface(fields=['filtered_file']),
                      name='outputspec')

    highpass.connect(inputspec, 'in_file', filter_, 'in_file')
    highpass.connect(inputspec, 'in_file', getsigma, 'in_file')
    highpass.connect(getsigma, 'op_string', filter_, 'op_string')
    highpass.connect(inputspec, 'in_file', meanfunc, 'in_file')
    highpass.connect(filter_, 'out_file', addmean, 'in_file')
    highpass.connect(meanfunc, 'out_file', addmean, 'operand_file')
    highpass.connect(addmean, 'out_file', outputspec, 'filtered_file')

    return highpass


def create_indnet_workflow(hp_cutoff=100, smoothing=5, threshold=0.5,
                           aggr_aroma=False, name="indnet"):
    indnet = Workflow(name=name)

    # Input node
    inputspec = Node(utility.IdentityInterface(fields=['t1_file',
                                                       'rs_file',
                                                       'templates',
                                                       'networks']),
                     name='inputspec')

    # T1 skullstrip
    t1_bet = Node(fsl.BET(), name="t1_bet")

    # Resting state preprocessing
    rs_featpreproc = create_featreg_preproc(highpass=False, whichvol='first',
                                            name='rs_featpreproc')
    rs_featpreproc.inputs.inputspec.fwhm = smoothing

    # Register resting state to MNI
    rs_2mni = create_reg_workflow(name='rs_2mni')
    rs_2mni.inputs.inputspec.target_image = fsl.Info.standard_image(
        'MNI152_T1_2mm.nii.gz')
    rs_2mni.inputs.inputspec.target_image_brain = fsl.Info.standard_image(
        'MNI152_T1_2mm_brain.nii.gz')
    rs_2mni.inputs.inputspec.config_file = 'T1_2_MNI152_2mm'

    # Create mask for ICA-AROMA
    rs_brainmask = Node(fsl.BET(), name='rs_brainmask')
    rs_brainmask.inputs.frac = 0.3
    rs_brainmask.inputs.mask = True
    rs_brainmask.inputs.no_output = True
    rs_brainmask.inputs.robust = True

    # Resting state ICA-AROMA
    rs_aroma = Node(fsl.ICA_AROMA(), name='rs_aroma')
    if aggr_aroma:
        rs_aroma.inputs.denoise_type = 'aggr'
    else:
        rs_aroma.inputs.denoise_type = 'nonaggr'

    # Resting state highpass filtering
    rs_highpass = create_highpass_filter(cutoff=hp_cutoff,
                                         name='rs_highpass')

    # Register T1 to MNI
    t1_2mni = Node(fsl.ApplyWarp(), name='t1_2mni')
    t1_2mni.inputs.ref_file = fsl.Info.standard_image(
        'MNI152_T1_2mm_brain.nii.gz')

    # Segment MNI T1
    t1_segmentation = Node(fsl.FAST(segments=True), name='t1_segmentation')

    # GM mask each network template
    gm_networktemplates = MapNode(fsl.ImageMaths(op_string='-mul'),
                                  iterfield=['in_file2'],
                                  name='gm_networktemplates')

    # Get first eigenvariate of each GM masked network template
    firsteigenvariates = MapNode(fsl.ImageMeants(show_all=True, eig=True),
                                 iterfield=['mask'],
                                 name='firsteigenvariates')

    # Get mean time course of WM signal
    wm_meansignal = Node(fsl.ImageMeants(), name='wm_meansignal')

    # Get mean time course of CSF signal
    csf_meansignal = Node(fsl.ImageMeants(), name='csf_meansignal')

    # Combine first eigenvariates and WM/CSF signals
    regressors = Node(utility.Merge(3), name='regressors')

    # z-transform regressors
    ztransform = MapNode(Ztransform(), iterfield=['in_file'], name='ztransform')

    # Create design matrix
    designmatrix = Node(DesignMatrix(), name='designmatrix')

    # Create contrasts
    contrasts = Node(Contrasts(), name='contrasts')

    # Dilate brain mask
    rs_brainmaskdilation = Node(fsl.DilateImage(), name='rs_brainmaskdilation')
    rs_brainmaskdilation.inputs.operation = 'max'

    # GLM
    glm = Node(fsl.GLM(), name='glm')
    glm.inputs.out_z_name = 'z_stats.nii.gz'
    glm.inputs.demean = True

    # Split z-maps
    zmaps = Node(fsl.Split(), name='zmaps')
    zmaps.inputs.dimension = 't'

    # Spatial Mixture Modelling
    smm = MapNode(fsl.SMM(),iterfield=['spatial_data_file'], name='smm')
    #smm.inputs.no_deactivation_class = True

    # Binarise results
    actmaps2binmasks = MapNode(
            fsl.ImageMaths(op_string='-thr {0} -bin'.format(threshold)),
            iterfield=['in_file'],
            name='actmaps2binmasks')

    # Split main maps from diff maps
    mainmaps = Node(SplitMaps(), name='mainmaps')

    # Combine diff maps
    exclusivemaps = MapNode(fsl.ImageMaths(),
                            iterfield=['in_file', 'op_string'],
                            name='exclusivemaps')

    # Rename main maps
    mainmaps_rename = MapNode(utility.Rename(), iterfield=['in_file',
                                                           'format_string'],
                              name='mainmaps_rename')
    mainmaps_rename.inputs.keep_ext = True

    # Rename exclusive  maps
    exclusivemaps_rename = MapNode(utility.Rename(),
                           iterfield=['in_file', 'format_string'],
                           name='exclusivemaps_rename')
    exclusivemaps_rename.inputs.keep_ext = True

    # Output node
    outputspec = Node(utility.IdentityInterface(fields=['mainfiles',
                                                        'exclusivefiles']),
                      name='outputspec')


    # Helper functions
    def get_first_item(x):
        return x[0]

    def get_second_item(x):
        return x[1]

    def get_third_item(x):
        return x[2]

    def get_components(x):
        return [y['components'] for y in x]

    def get_name(x):
        return [y['name'] for y in x]


    # Connect everything
    indnet.connect(inputspec, 'rs_file', rs_featpreproc, 'inputspec.func')
    indnet.connect(rs_featpreproc, 'outputspec.smoothed_files',
                   rs_2mni, 'inputspec.source_files')
    indnet.connect(inputspec, 't1_file',
               rs_2mni, 'inputspec.anatomical_image')
    indnet.connect(rs_featpreproc, 'outputspec.reference',
               rs_2mni, 'inputspec.mean_image')
    indnet.connect(rs_2mni, ('outputspec.transformed_files', get_first_item),
               rs_aroma, 'in_file')
    indnet.connect(rs_featpreproc, ('outputspec.motion_parameters',
                                    get_first_item),
               rs_aroma, 'motion_parameters')
    indnet.connect(rs_2mni, 'outputspec.transformed_mean',
               rs_brainmask, 'in_file')
    indnet.connect(rs_brainmask,'mask_file', rs_aroma, 'mask')
    if aggr_aroma:
        indnet.connect(rs_aroma, 'aggr_denoised_file',
                rs_highpass, 'inputspec.in_file')
    else:
        indnet.connect(rs_aroma, 'nonaggr_denoised_file',
                rs_highpass, 'inputspec.in_file')
    indnet.connect(inputspec, 't1_file', t1_bet, 'in_file')
    indnet.connect(t1_bet, 'out_file', t1_2mni, 'in_file')
    indnet.connect(rs_2mni, 'outputspec.anat2target_transform',
               t1_2mni, 'field_file')
    indnet.connect(t1_2mni, 'out_file', t1_segmentation, 'in_files')
    indnet.connect(t1_segmentation, ('tissue_class_files', get_second_item),
               gm_networktemplates, 'in_file')
    indnet.connect(inputspec, 'templates',
               gm_networktemplates, 'in_file2')
    indnet.connect(rs_highpass, 'outputspec.filtered_file',
               firsteigenvariates, 'in_file')
    indnet.connect(gm_networktemplates, 'out_file',
               firsteigenvariates, 'mask')
    indnet.connect(t1_segmentation, ('tissue_class_files', get_third_item),
               wm_meansignal, 'mask')
    indnet.connect(rs_highpass, 'outputspec.filtered_file',
                   wm_meansignal, 'in_file')
    indnet.connect(t1_segmentation, ('tissue_class_files',
                   get_first_item), csf_meansignal, 'mask')
    indnet.connect(rs_highpass, 'outputspec.filtered_file',
                   csf_meansignal, 'in_file')
    indnet.connect(firsteigenvariates, 'out_file', regressors,'in1')
    indnet.connect(wm_meansignal, 'out_file', regressors, 'in2')
    indnet.connect(csf_meansignal, 'out_file', regressors, 'in3')
    indnet.connect(regressors, 'out', ztransform, 'in_file')
    indnet.connect(ztransform, 'out_file', designmatrix, 'in_files')
    indnet.connect(inputspec, ('networks', get_components),
                   contrasts, 'in_list')
    indnet.connect(designmatrix, 'out_file', glm, 'design')
    indnet.connect(designmatrix, 'out_file', contrasts, 'design')
    indnet.connect(contrasts, 'out_file', glm, 'contrasts')
    indnet.connect(rs_brainmask, 'mask_file', glm, 'mask')
    indnet.connect(rs_highpass, 'outputspec.filtered_file', glm, 'in_file')
    indnet.connect(glm, 'out_z', zmaps, 'in_file')
    indnet.connect(zmaps, 'out_files', smm, 'spatial_data_file')
    indnet.connect(rs_brainmask, 'mask_file', smm, 'mask')
    indnet.connect(smm, 'activation_p_map', actmaps2binmasks, 'in_file')
    indnet.connect(actmaps2binmasks, 'out_file', mainmaps, 'in_files')
    indnet.connect(inputspec, 'networks', mainmaps, 'in_networks')
    indnet.connect(mainmaps, 'out_mains', mainmaps_rename, 'in_file')
    indnet.connect(inputspec, ('networks', get_name),
                   mainmaps_rename, 'format_string')
    indnet.connect(mainmaps_rename, 'out_file', outputspec, 'mainfiles')
    indnet.connect(mainmaps, 'out_firsts', exclusivemaps, 'in_file')
    indnet.connect(mainmaps, 'out_opstrings', exclusivemaps, 'op_string')
    indnet.connect(exclusivemaps, 'out_file',
                   exclusivemaps_rename, 'in_file')
    indnet.connect(inputspec, ('networks', get_name),
                   exclusivemaps_rename, 'format_string')
    indnet.connect(exclusivemaps_rename, 'out_file',
                   outputspec, 'exclusivefiles')

    return indnet
