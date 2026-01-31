import abagen
import pandas
data_dir = '/path/to/abagen_processing/AHBA_data'
atlas_file = '/data/liwei/model/abagen_processing/aal.nii/aal.nii'
expression = abagen.get_expression_data(atlas_file, missing='centroids',data_dir=data_dir)
expression.to_csv('/path/to/abagen_processing/noAAL_result/01aal_result.csv',index=False,sep=',')


data_dir = '/path/to/abagen_processing/AHBA_data'
atlas = abagen.fetch_desikan_killiany()
expression = abagen.get_expression_data(atlas['image'], atlas['info'], missing='centroids',data_dir=data_dir)
expression.to_csv('/path/to/abagen_processing/noAAL_result/dk_result.csv',index=False,sep=',')
import torch
data_dir = '/path/to/abagen_processing/AHBA_data'
expression, coords = abagen.get_samples_in_mask(mask=None,
data_dir=data_dir)
expression.to_csv('/path/to/abagen_processing/noAAL_result/noAAL_result.csv',index=False,sep=',')

expression, coords = abagen.get_samples_in_mask(mask=None,
probe_selection='max_intensity')
expression.to_csv('/path/to/abagen_processing/noAAL_result/noAAL_result_v0.csv',index=False,sep=',')
print("end")

expression, coords = abagen.get_samples_in_mask(mask=None,
probe_selection='max_intensity',
sample_norm=None,
gene_norm=None,
lr_mirror='bidirectional')
expression.to_csv('/path/to/abagen_processing/noAAL_result/noAAL_result_v1.csv',index=False,sep=',')
print("end")
import numpy as np
expression, coords = abagen.get_samples_in_mask(mask=None,
probe_selection='max_intensity',
sample_norm=None,
gene_norm=None)
expression.to_csv('/path/to/abagen_processing/noAAL_result/noAAL_result_nobiaozhunhua.csv', index=False, sep=',')
np.savetxt('/path/to/abagen_processing/noAAL_result/noAAL_coords_nobiaozhunhua.csv', coords, delimiter=',')
print("end")
import numpy as np
expression, coords = abagen.get_samples_in_mask(mask=None,
probe_selection='max_intensity',
data_dir=data_dir)
expression.to_csv('/path/to/abagen_processing/result/noAAL_result_v3.csv', index=False, sep=',')
np.savetxt('/path/to/abagen_processing/noAAL_result/noAAL_coords_v3.csv', coords, delimiter=',')
print("end")
import pandas as pd
import numpy as np
from mni_to_atlas import AtlasBrowser
file_path = '/path/to/abagen_processing/noAAL_result/noAAL_coords_nobiaozhunhua.csv'
coordinates_df = pd.read_csv(file_path, header=None, names=['mni_x', 'mni_y', 'mni_z'])
coordinates_df = coordinates_df.apply(pd.to_numeric, errors='coerce')
coordinates_df = coordinates_df.dropna()
coordinates = coordinates_df[['mni_x', 'mni_y', 'mni_z']].values
atlas_browser = AtlasBrowser(atlas='AAL3')
regions = atlas_browser.find_regions(coordinates)
coordinates_df['aal.label'] = regions
output_file_path = '/path/to/abagen_processing/noAAL_result_with_regions/noAAL3_coords_nobiaozhunhua_with_regions_Python.csv'
coordinates_df.to_csv(output_file_path, index=False, header=True)
from mni_to_atlas import AtlasBrowser
file_path = '/path/to/abagen_processing/noAAL_result/noAAL_coords_nobiaozhunhua.csv'
coordinates_df = pd.read_csv(file_path, header=None, names=['mni_x', 'mni_y', 'mni_z'])
coordinates_df = coordinates_df.apply(pd.to_numeric, errors='coerce')
coordinates_df = coordinates_df.dropna()
coordinates = coordinates_df[['mni_x', 'mni_y', 'mni_z']].values
atlas_browser = AtlasBrowser(atlas='AAL')
regions = atlas_browser.find_regions(coordinates)
coordinates_df['aal.label'] = regions
output_file_path = '/path/to/abagen_processing/noAAL_result_with_regions/noAAL_coords_nobiaozhunhua_with_regions_Python.csv'
coordinates_df.to_csv(output_file_path, index=False, header=True)
