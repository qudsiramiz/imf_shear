from imf_shear import __version__


def test_version():
    assert __version__ == '0.1.0'

from imf_shear import shear_angle_calculator as sac

class TestIMFShear:
    _kwargs_default = {
        "b_imf" : [-5, 1, -1],
        "np_imf" : 5,
        "v_imf" : [-450, 50, 50],
        "min_max_val" : 15,
        "dmp" : 0.5,
        "dr" : None,
        "model_type" : "t96",
        "angle_units" : "degrees",
        "use_real_data" : False,
        "time_observation" : None,
        "dt" : 5,
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
    