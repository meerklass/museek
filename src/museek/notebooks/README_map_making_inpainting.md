Notebook Usage Guide for `map_making_inpainting.ipynb`

This notebook performs inpainting and map making for each single MeerKLASS block. 

1. Environment Dependency:
Same as MuSEEK pipeline, check https://github.com/meerklass/museek

2. Running the Notebook:
check the Notebooks Section in  https://github.com/meerklass/museek

3. Input Data
The notebook expects MeerKLASS calibrated data from the aoflagger postcalibration plugin of MuSEEK

4. Parameters and Their Meanings
Below is a description of the main parameters defined in the configuration section of the notebook.

configuration example:
```python
BOX = 'BOX3'
input_path = '/idia/projects/meerklass/MEERKLASS-1/uhf_data/OT2024/pipeline/Pipeline_PS_AOFlagger_test/'+BOX+'/'
output_path = '/idia/users/wkhu/calibration_results/'+BOX+'/maps/'
data_name = 'aoflagger_plugin_postcalibration.pickle'
block_name = '1710186924'
map_version = '_inpainting'  ### the version (selfcali or not) of the map, 
                             ### if running notebook for selfcali data, use map_version = '_selfcali_inpainting'
#map_version = '_selfcali_inpainting' 

threshold_MHz = 30.  #[MHz] if a long continuous frequency region is masked, this timestamp will be totally masked 
inpainting_window=20 #[MHz] inpainting the masked regions by fitting a polynomial using +-inpainting_window 
                     #      of the unmasked data around the masked regions
inpainting_polydeg=6 # the degree of polynomials fit in inpainting

mask_antnum_threshold=10 #  masking the pixels where <=mask_antnum_threshold antennas contributes

pix_reso = 0.5   # map resolution [deg]
x_crval = 190.   # map center x(ra) [deg]
y_crval = -1.5   # map center y(dec) [deg]
x_min = x_crval-30.  # map region x(ra)
x_max = x_crval+30.  # map region x(ra)
y_min = y_crval-10.  # map region y(dec)
y_max = y_crval+10.  # map region y(dec)
```

5. Contact

For questions or issues, contact:
Wenkai Hu, wkhu@nao.cas.cn

