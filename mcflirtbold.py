import os
os.environ['FSLOUTPUTTYPE'] = 'NIFTI'
os.environ['FSLDIR'] = '/Users/mac/fsl/share/fsl'
os.environ['PATH'] = os.environ['FSLDIR'] + '/bin:' + os.environ['PATH']

from nipype.interfaces import fsl

mcflt = fsl.MCFLIRT()

mcflt.inputs.in_file = 'sub-01_ses-01_bold.nii'
mcflt.inputs.out_file = 'sub-01_ses-01_bold_mcf.nii'
mcflt.inputs.save_plots = True
mcflt.inputs.save_mats = True

res = mcflt.run()
print(res.outputs)