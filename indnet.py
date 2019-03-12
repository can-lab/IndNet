"""IndNet.

A Nipype implementation of a dual-regression-like seed-based approach to
individualize general binary templates to specific subjects.

"""

__author__ = "Florian Krause <f.krause@donders.ru.nl>, \
              Nikos Kogias <n.kogias@student.ru.nl>"

__version__ = '0.2.0'
__date__ = '2018-10-31'


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
    in_file = File(exists=True,
                   desc='textfile with values to be z-transformed',
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
        traits.Any, mandatory=True,
        desc='list of one-column regressors textfiles')


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


def create_segments_2func_workflow(threshold= 0.5,
                                  name='segments_2func_workflow'):
    segments_2func_workflow = Workflow(name=name)

    # Input Node
    inputspec = Node(utility.IdentityInterface(fields=['segments', 
                                                       'premat', 
                                                       'func_file']), 
                     name= 'inputspec')

    # Calculate inverse matrix of EPI to T1
    anat_2func_matrix = Node(fsl.ConvertXFM(invert_xfm= True), 
                             name= 'anat_2func_matrix')

    # Transform segments to EPI space
    segments_2func_apply = MapNode(fsl.ApplyXFM(), iterfield= ['in_file'], 
                                  name='segments_2func_apply')

    # Threshold segments
    segments_threshold = MapNode(fsl.ImageMaths(
            op_string= '-thr {0} -bin'.format(threshold)), 
            iterfield= ['in_file'], 
            name= 'segments_threshold')

    # Output Node
    outputspec = Node(utility.IdentityInterface(
        fields=['segments_2func_files', 'anat_2func_matrix_file']), 
        name= 'outputspec')

    segments_2func_workflow.connect(inputspec, 'premat', 
                                    anat_2func_matrix, 'in_file')
    segments_2func_workflow.connect(inputspec, 'segments', 
                                    segments_2func_apply, 'in_file')
    segments_2func_workflow.connect(inputspec, 'func_file', 
                                    segments_2func_apply, 'reference')
    segments_2func_workflow.connect(anat_2func_matrix, 'out_file', 
                                    segments_2func_apply, 'in_matrix_file')
    segments_2func_workflow.connect(segments_2func_apply, 'out_file', 
                                    segments_threshold, 'in_file')    
    segments_2func_workflow.connect(anat_2func_matrix, 'out_file', 
                                    outputspec, 'anat_2func_matrix_file')
    segments_2func_workflow.connect(segments_threshold, 'out_file', 
                                    outputspec, 'segments_2func_files')

    return segments_2func_workflow


def create_templates_2func_workflow(threshold= 0.5, 
                                   name= 'templates_2func_workflow'):
    templates_2func_workflow = Workflow(name= name)

    # Input Node
    inputspec = Node(utility.IdentityInterface(fields= ['func_file', 
                                                        'premat', 
                                                        'warp', 
                                                        'templates',]), 
                     name= 'inputspec')

    # Get the overal EPI to MNI warp
    func_2mni_warp = Node(fsl.ConvertWarp(), name='func_2mni_warp')
    func_2mni_warp.inputs.reference = fsl.Info.standard_image(
            'MNI152_T1_2mm.nii.gz') 

    # Calculate the inverse warp
    mni_2func_warp = Node(fsl.InvWarp(), name= 'mni_2func_warp')

    # Transform MNI templates to EPI space
    templates_2func_apply = MapNode(fsl.ApplyWarp(), iterfield= ['in_file'], 
                                   name= 'templates_2func_apply')

    # Threshold templates
    templates_threshold = MapNode(fsl.ImageMaths(
            op_string= '-thr {0} -bin'.format(threshold)), 
            iterfield= ['in_file'], 
            name= 'templates_threshold')

    # Output Node
    outputspec = Node(utility.IdentityInterface(
        fields=['templates_2func_files', 'func_2mni_warp']), 
        name= 'outputspec')

    # Connect the workflow nodes
    templates_2func_workflow.connect(inputspec, 'premat', 
                                    func_2mni_warp, 'premat')
    templates_2func_workflow.connect(inputspec, 'warp', 
                                    func_2mni_warp, 'warp1')
    templates_2func_workflow.connect(inputspec, 'func_file', 
                                    mni_2func_warp, 'reference')
    templates_2func_workflow.connect(func_2mni_warp, 'out_file', 
                                    mni_2func_warp, 'warp')
    templates_2func_workflow.connect(inputspec, 'templates', 
                                    templates_2func_apply, 'in_file')
    templates_2func_workflow.connect(inputspec, 'func_file', 
                                    templates_2func_apply, 'ref_file')
    templates_2func_workflow.connect(mni_2func_warp, 'inverse_warp', 
                                    templates_2func_apply, 'field_file')
    templates_2func_workflow.connect(templates_2func_apply, 'out_file', 
                                    templates_threshold, 'in_file')
    templates_2func_workflow.connect(func_2mni_warp, 'out_file', 
                                    outputspec, 'func_2mni_warp')
    templates_2func_workflow.connect(templates_threshold, 'out_file', 
                                    outputspec, 'templates_2func_files')

    return templates_2func_workflow


def create_network_masks_workflow(name="network_masks", smm_threshold=0.5):

    network_masks = Workflow(name=name)

    # Input node
    inputspec = Node(utility.IdentityInterface(fields=['actmaps',
                                                       'networks']),
                     name='inputspec')

    # Binarise results
    actmaps2binmasks = MapNode(
            fsl.ImageMaths(op_string='-thr {0} -bin'.format(smm_threshold)),
            iterfield=['in_file'],
            name='actmaps2binmasks')

    # Split main masks from exclusive masks
    mainmasks = Node(SplitMaps(), name='mainmasks')

    # Combine exclusive masks
    exclusivemasks = MapNode(fsl.ImageMaths(), iterfield=['in_file',
                                                          'op_string'],
                             name='exclusivemasks')

    # Rename main masks
    mainmasks_rename = MapNode(utility.Rename(), 
                             iterfield=['in_file', 'format_string'],
                               name='mainmasks_rename')
    mainmasks_rename.inputs.keep_ext = True

    # Rename exclusive masks
    exclusivemasks_rename = MapNode(utility.Rename(),
                                    iterfield=['in_file', 'format_string'],
                                    name='exclusivemasks_rename')
    exclusivemasks_rename.inputs.keep_ext = True

    # Output Node
    outputspec = Node(utility.IdentityInterface(fields=['main_masks', 
                                                        'exclusive_masks']), 
                                                name= 'outputspec')

    # Helper functions

    def get_names(x):
        return [y['name'] for y in x]

    network_masks.connect(inputspec, 'actmaps', actmaps2binmasks, 'in_file')
    network_masks.connect(actmaps2binmasks, 'out_file', mainmasks, 'in_files')
    network_masks.connect(inputspec, 'networks', mainmasks, 'in_networks')
    network_masks.connect(mainmasks, 'out_mains', mainmasks_rename, 'in_file')
    network_masks.connect(inputspec, ('networks', get_names),
                         mainmasks_rename, 'format_string')
    network_masks.connect(mainmasks, 'out_firsts', exclusivemasks, 'in_file')
    network_masks.connect(mainmasks, 'out_opstrings',
                          exclusivemasks, 'op_string')
    network_masks.connect(exclusivemasks, 'out_file',
                         exclusivemasks_rename, 'in_file')
    network_masks.connect(inputspec, ('networks', get_names),
                         exclusivemasks_rename, 'format_string')
    network_masks.connect(mainmasks_rename, 'out_file',
                          outputspec, 'main_masks')
    network_masks.connect(exclusivemasks_rename, 'out_file',
                          outputspec, 'exclusive_masks')

    return network_masks


def create_indnet_workflow(hp_cutoff=100, smoothing=5, 
                           smm_threshold=0.5, 
                           binarise_threshold=0.5, 
                           melodic_seed=None,  
                           aggr_aroma=False, name="indnet"):

    indnet = Workflow(name=name)

    # Input node
    inputspec = Node(utility.IdentityInterface(fields=['anat_file',
                                                       'func_file',
                                                       'templates',
                                                       'networks']),
                     name='inputspec')

    # T1 skullstrip
    anat_bet = Node(fsl.BET(), name= "anat_bet")

    # EPI preprocessing
    func_realignsmooth = create_featreg_preproc(highpass= False,
                                               whichvol= 'first', 
                                               name= 'func_realignsmooth')
    func_realignsmooth.inputs.inputspec.fwhm = smoothing

    # Transform EPI to MNI space
    func_2mni = create_reg_workflow(name= 'func_2mni')
    func_2mni.inputs.inputspec.target_image = fsl.Info.standard_image(
            'MNI152_T1_2mm.nii.gz')
    func_2mni.inputs.inputspec.target_image_brain = fsl.Info.standard_image(
            'MNI152_T1_2mm_brain.nii.gz')
    func_2mni.inputs.inputspec.config_file = 'T1_2_MNI152_2mm'

    # Segmentation of T1
    anat_segmentation = Node(fsl.FAST(output_biascorrected=True), 
                           name= 'anat_segmentation')

    # Transfrom segments to EPI space
    segments_2func= create_segments_2func_workflow(
            threshold=binarise_threshold, name= 'segments_2func')

    # Transform templates to EPI space
    templates_2func= create_templates_2func_workflow(
            threshold=binarise_threshold, name= 'templates_2func')

    # Mask network templates with GM
    gm_mask_templates = MapNode(fsl.ImageMaths(op_string= '-mul'),
                                iterfield= ['in_file2'],
                                name= 'gm_mask_templates')

    # Mask for ICA-AROMA and statistics
    func_brainmask = Node(fsl.BET(frac= 0.3, mask= True, 
                                 no_output= True, robust= True), 
                         name= 'func_brainmask')

    # Melodic ICA
    if melodic_seed != None:
        func_melodic = Node(fsl.MELODIC(args= '--seed={}'.format(melodic_seed), out_stats= True), 
                                        name= 'func_melodic')

    # ICA-AROMA
    func_aroma = Node(fsl.ICA_AROMA(), name= 'func_aroma')
    if aggr_aroma:
        func_aroma.inputs.denoise_type = 'aggr'

    else:
        func_aroma.inputs.denoise_type = 'nonaggr' 

    # Highpass filter ICA results
    func_highpass= create_highpass_filter(cutoff= hp_cutoff,
                                         name= 'func_highpass')

    # Calculate mean CSF sgnal
    csf_meansignal = Node(fsl.ImageMeants(), name= 'csf_meansignal')

    # Calculate mean WM signal
    wm_meansignal = Node(fsl.ImageMeants(), name= 'wm_meansignal')

    # Calculate first Eigenvariates 
    firsteigenvariates = MapNode(fsl.ImageMeants(show_all= True, eig= True),
                                 iterfield= ['mask'],
                                 name= 'firsteigenvariates')

    # Combine first eigenvariates and wm/csf signals
    regressors = Node(utility.Merge(3), name='regressors')

    # z-transform regressors
    ztransform = MapNode(Ztransform(), iterfield=['in_file'],
                         name='ztransform')

    # Create design matrix
    designmatrix = Node(DesignMatrix(), name='designmatrix')

    # Create contrasts
    contrasts = Node(Contrasts(), name='contrasts')

    # GLM
    glm = Node(fsl.GLM(), name='glm')
    glm.inputs.out_z_name = 'z_stats.nii.gz'
    glm.inputs.demean = True

    # Split z-maps
    zmaps = Node(fsl.Split(), name='zmaps')
    zmaps.inputs.dimension = 't'

    # Spatial Mixture Modelling
    smm = MapNode(fsl.SMM(),iterfield=['spatial_data_file'], name='smm') 

    # Transform probability maps to native (anat) space
    actmaps_2anat= MapNode(fsl.ApplyXFM(), iterfield=['in_file'], 
                         name='actmaps_2anat') 

    # Transform probability maps to MNI space
    actmaps_2mni= MapNode(fsl.ApplyWarp(), iterfield=['in_file'], 
                        name='actmaps_2mni')  
    actmaps_2mni.inputs.ref_file = fsl.Info.standard_image(
            'MNI152_T1_2mm.nii.gz')

    # Create network masks in native (func) space
    network_masks_func = create_network_masks_workflow(
            name='network_masks_func', smm_threshold=smm_threshold)

    # Create network masks in native (anat) space
    network_masks_anat = create_network_masks_workflow(
            name='network_masks_anat', smm_threshold=smm_threshold)

    # Create network masks in MNI space
    network_masks_mni = create_network_masks_workflow(
            name='network_masks_mni', smm_threshold=smm_threshold)

    # Output node
    outputspec = Node(utility.IdentityInterface(
        fields=['network_masks_func_main',
                'network_masks_func_exclusive',
                'network_masks_anat_main',
                'network_masks_anat_exclusive',
                'network_masks_mni_main',
                'network_masks_mni_exclusive',
                'preprocessed_func_file',
                'preprocessed_anat_file',
                'motion_parameters',
                'func2anat_transform',
                'anat2mni_transform']),
        name='outputspec')


    # Helper functions
    def get_first_item(x):
        try:
            return x[0]
        except:
            return x

    def get_second_item(x):
        return x[1]

    def get_third_item(x):
        return x[2]

    def get_components(x):
        return [y['components'] for y in x]


    # Connect the nodes

    # anat_bet
    indnet.connect(inputspec, 'anat_file', anat_bet, 'in_file')

    # func_realignsmooth
    indnet.connect(inputspec, 'func_file',
                   func_realignsmooth, 'inputspec.func')

    # func_2mni
    indnet.connect(func_realignsmooth, ('outputspec.smoothed_files',
                                        get_first_item), 
                   func_2mni, 'inputspec.source_files')
    indnet.connect(inputspec, 'anat_file', 
                   func_2mni, 'inputspec.anatomical_image')
    indnet.connect(func_realignsmooth, 'outputspec.reference', 
                   func_2mni, 'inputspec.mean_image')

    # anat_segmentation
    indnet.connect(anat_bet, 'out_file', anat_segmentation, 'in_files')

    # segments_2func
    indnet.connect(anat_segmentation, 'partial_volume_files', 
                   segments_2func, 'inputspec.segments' )
    indnet.connect(func_2mni, 'outputspec.func2anat_transform', 
                   segments_2func, 'inputspec.premat')
    indnet.connect(func_realignsmooth, ('outputspec.smoothed_files',
                                        get_first_item),
                   segments_2func, 'inputspec.func_file')

    # templates_2func
    indnet.connect(func_realignsmooth, ('outputspec.smoothed_files',
                                        get_first_item),
                   templates_2func, 'inputspec.func_file')
    indnet.connect(func_2mni, 'outputspec.func2anat_transform', 
                   templates_2func, 'inputspec.premat')
    indnet.connect(func_2mni, 'outputspec.anat2target_transform', 
                   templates_2func, 'inputspec.warp')
    indnet.connect(inputspec, 'templates', 
                   templates_2func, 'inputspec.templates')

    # gm_mask_templates
    indnet.connect(segments_2func, ('outputspec.segments_2func_files',
                                   get_second_item), 
                   gm_mask_templates, 'in_file') 
    indnet.connect(templates_2func, 'outputspec.templates_2func_files', 
                   gm_mask_templates, 'in_file2')

    # func_brainmask
    indnet.connect(func_realignsmooth, ('outputspec.smoothed_files',
                                        get_first_item),
                   func_brainmask, 'in_file')

    # func_melodic
    if melodic_seed != None:    
        indnet.connect(func_realignsmooth, ('outputspec.smoothed_files',
                                            get_first_item),
                        func_melodic, 'in_files')
        indnet.connect(func_brainmask, 'mask_file', func_melodic, 'mask')


    # func_aroma
    indnet.connect(func_realignsmooth, ('outputspec.smoothed_files',
                                        get_first_item),
                   func_aroma, 'in_file')
    indnet.connect(func_2mni, 'outputspec.func2anat_transform',
                   func_aroma, 'mat_file')
    indnet.connect(func_2mni, 'outputspec.anat2target_transform',
                   func_aroma, 'fnirt_warp_file')
    indnet.connect(func_realignsmooth, ('outputspec.motion_parameters',
                                        get_first_item),
                   func_aroma, 'motion_parameters')
    indnet.connect(func_brainmask, 'mask_file', func_aroma, 'mask')
    if melodic_seed != None:
        indnet.connect(func_melodic, 'out_dir', func_aroma, 'melodic_dir')


    # func_highpass
    if aggr_aroma:
        indnet.connect(func_aroma, 'aggr_denoised_file', 
                       func_highpass, 'inputspec.in_file')
    else:
        indnet.connect(func_aroma, 'nonaggr_denoised_file', 
                       func_highpass, 'inputspec.in_file')

    # csf_meansignal
    indnet.connect(segments_2func, ('outputspec.segments_2func_files',
                                   get_first_item),
                   csf_meansignal, 'mask')
    indnet.connect(func_highpass, 'outputspec.filtered_file', 
                   csf_meansignal, 'in_file' )

    # wm_meansignal
    indnet.connect(segments_2func, ('outputspec.segments_2func_files',
                                   get_third_item),
                   wm_meansignal, 'mask')
    indnet.connect(func_highpass, 'outputspec.filtered_file', 
                   wm_meansignal, 'in_file' )

    # firsteigenvariates
    indnet.connect(gm_mask_templates, 'out_file', 
                   firsteigenvariates, 'mask')
    indnet.connect(func_highpass, 'outputspec.filtered_file', 
                   firsteigenvariates, 'in_file' )
    
    # regressors
    indnet.connect(firsteigenvariates, 'out_file', regressors,'in1')
    indnet.connect(wm_meansignal, 'out_file', regressors, 'in2')
    indnet.connect(csf_meansignal, 'out_file', regressors, 'in3')

    # ztransform
    indnet.connect(regressors, 'out', ztransform, 'in_file')

    # designmatrix
    indnet.connect(ztransform, 'out_file', designmatrix, 'in_files')

    # contrasts
    indnet.connect(inputspec, ('networks', get_components),
                   contrasts, 'in_list')
    indnet.connect(designmatrix, 'out_file', glm, 'design')
    indnet.connect(designmatrix, 'out_file', contrasts, 'design')

    # glm
    indnet.connect(contrasts, 'out_file', glm, 'contrasts')
    indnet.connect(func_brainmask, 'mask_file', glm, 'mask')
    indnet.connect(func_highpass, 'outputspec.filtered_file', glm, 'in_file')

    # zmaps
    indnet.connect(glm, 'out_z', zmaps, 'in_file')

    # smm
    indnet.connect(zmaps, 'out_files', smm, 'spatial_data_file')
    indnet.connect(func_brainmask, 'mask_file', smm, 'mask')

    # actmaps_2anat
    indnet.connect(smm, 'activation_p_map', actmaps_2anat, 'in_file')
    indnet.connect(func_2mni, 'outputspec.func2anat_transform', 
                   actmaps_2anat, 'in_matrix_file')
    indnet.connect(anat_bet, 'out_file', actmaps_2anat, 'reference')

    # actmaps_2mni
    indnet.connect(smm, 'activation_p_map', actmaps_2mni, 'in_file')
    indnet.connect(templates_2func, 'outputspec.func_2mni_warp', 
                   actmaps_2mni, 'field_file' )

    # network_masks_func
    indnet.connect(smm, 'activation_p_map',
                   network_masks_func, 'inputspec.actmaps')
    indnet.connect(inputspec, 'networks',
                   network_masks_func, 'inputspec.networks')

    # network_masks_anat
    indnet.connect(actmaps_2anat, 'out_file',
                   network_masks_anat, 'inputspec.actmaps')
    indnet.connect(inputspec, 'networks',
                   network_masks_anat, 'inputspec.networks')

    # network_masks_mni
    indnet.connect(actmaps_2mni, 'out_file',
                   network_masks_mni, 'inputspec.actmaps')
    indnet.connect(inputspec, 'networks',
                   network_masks_mni, 'inputspec.networks')

    # output node
    indnet.connect(network_masks_func, 'outputspec.main_masks', 
                   outputspec, 'network_masks_func_main')
    indnet.connect(network_masks_func, 'outputspec.exclusive_masks',
                   outputspec, 'network_masks_func_exclusive')
    indnet.connect(network_masks_anat, 'outputspec.main_masks', 
                   outputspec, 'network_masks_anat_main')
    indnet.connect(network_masks_anat, 'outputspec.exclusive_masks',
                   outputspec, 'network_masks_anat_exclusive')
    indnet.connect(network_masks_mni, 'outputspec.main_masks', 
                   outputspec, 'network_masks_mni_main')
    indnet.connect(network_masks_mni, 'outputspec.exclusive_masks', 
                   outputspec, 'network_masks_mni_exclusive')
    indnet.connect(func_highpass, 'outputspec.filtered_file',
                   outputspec, 'preprocessed_func_file')
    indnet.connect(anat_segmentation, 'restored_image', 
                   outputspec, 'preprocessed_anat_file')
    indnet.connect(func_realignsmooth, ('outputspec.motion_parameters',
                                        get_first_item), 
                   outputspec, 'motion_parameters')
    indnet.connect(func_2mni, 'outputspec.func2anat_transform', 
                   outputspec, 'func2anat_transform')
    indnet.connect(func_2mni, 'outputspec.anat2target_transform', 
                   outputspec, 'anat2mni_transform')

    return indnet
