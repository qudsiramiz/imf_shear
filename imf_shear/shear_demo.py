import importlib
import shear_angle_calculator as sac
importlib.reload(sac)

inputs = {
    "b_imf" : None,
    "np_imf" : None,
    "v_imf" : None,
    "min_max_val" : 20,
    "dmp" : None,
    "dr" : None,
    "model_type" : "t96",
    "angle_units" : "degrees",
    "use_real_data" : True,
    "time_observation" : '2015-09-11 15:23:00',
    "dt" : 1,
    "save_data" : True,
    "data_file" : None,
    "plot_figure" : True,
    "clip_image" : False,
    "save_figure" : True,
    "figure_size" : (6,6),
    "figure_file" : "shear_angle_calculator",
    "figure_format" : "pdf",
    "verbose" : True
}

shear = sac.shear_angle_calculator(**inputs)
#time_observation = '2020-10-01 14:03:06'
#
## Remove spaces from time_observation
#time_observation = time_observation.replace(" ", "")