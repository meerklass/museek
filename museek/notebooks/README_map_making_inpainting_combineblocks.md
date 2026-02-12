Notebook Usage Guide for `map_making_inpainting_combineblocks.ipynb`

This notebook is for combining the inpainted blocks for each single box (combining blocks), and creating the model. 

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
block_name_list = ['1710186924', '1710878483', '1712085987', '1713047732', '1721666164', '1710272607', 
                   '1710964969', '1712172417', '1713133906', '1721751682', '1710358887', '1711223036',  
                   '1713222040', '1722185106', '1710533408', '1711395125', '1713306983', '1723046486', 
                   '1710618377', '1712015185', '1712685146', '1714428474', '1710703585', '1712791459', 
                   '1716498372'] ## BOX3

bad_block_list = ['1711395125', '1712015185', '1711223036', '1713306983', '1713222040']  ## bad blocks 
block_process_list = [block_name for block_name in block_name_list if block_name not in bad_block_list]  ## excluding bad blocks

BOX = 'BOX3'
input_path = '/idia/users/wkhu/calibration_results/'+BOX+'/maps/'
output_path = '/idia/users/wkhu/calibration_results/'+BOX+'/maps/'

model_polyfit_deg = 6  ### polynomial degree for model fitting

map_version = '_inpainting'  ### the version (selfcali or not) of the map   
#map_version = '_selfcali_inpainting' 
```

5. Contact

For questions or issues, contact:
Wenkai Hu, wkhu@nao.cas.cn

