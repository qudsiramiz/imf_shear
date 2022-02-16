import calendar
import datetime
import multiprocessing as mp
import os
import warnings

import h5py as hf
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import scipy as sp
from dateutil import parser
from matplotlib import widgets
from matplotlib.pyplot import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.filters import frangi, hessian, meijering, sato
from tabulate import tabulate

# Set the fontstyle to Times New Roman
font = {'family': 'serif', 'weight': 'normal', 'size': 10}
plt.rc('font', **font)
plt.rc('text', usetex=True)


def get_shear(b_vec_1, b_vec_2, angle_units="radians"):
    r"""
    Get the shear angle between two magnetic field lines.

    Parameters
    ----------
    b_vec_1 : array of shape 1x3
        Input magnetic field vector.
    b_vec_2 : array of shape 1x3
        Input magnetic field vector.
    angle_unit : str, optional
        Preferred unit of angle returned by the code. Default is "radians".

    Raises
    ------
    KeyError If the key input_angle is not set to "radians" or "degrees" then the code raises
        a key error.

    Returns
    -------
    angle: float
        Angle between the two vectors, in radians by default
    """
    unit_vec_1 = b_vec_1/np.linalg.norm(b_vec_1)
    unit_vec_2 = b_vec_2/np.linalg.norm(b_vec_2)
    angle = np.arccos(np.dot(unit_vec_1, unit_vec_2))

    if (angle_units == "radians"):
        return angle
    elif (angle_units == "degrees"):
        return angle * 180/np.pi
    else:
        raise KeyError("angle_unit must be radians or degrees")


def get_sw_params(
    omni_level="hro",
    time_clip=True,
    trange=None,
    verbose=False
    ):
    r"""
    Get the solar wind parameters from the OMNI database.

    Parameters
    ----------
    probe : str
        The probe to use. Default is 'None'.
    omni_level : str
        The omni data level to use. Options are 'hro' and 'hro2'. Default is 'hro'.
    time_clip : bool
        If True, the data will be clipped to the time range specified by trange. Default is True.
    trange : list or an array of length 2
        The time range to use. Should in the format [start, end], where start and end times should
        be a string in the format 'YYYY-MM-DD HH:MM:SS'.
    mms_probe_num : str
        The MMS probe to use. Options are '1', '2', '3' and '4'. Default is None.
    verbose : bool
        If True, print out a few messages and the solar wind parameters. Default is False.

    Raises
    ------
    ValueError: If the probe is not one of the options.
    ValueError: If the trange is not in the correct format.

    Returns
    -------
    sw_params : dict
        The solar wind parameters.
    """

    import pyspedas as spd
    import pytplot as ptt
    import geopack.geopack as gp
    if trange is None:
        raise ValueError("trange must be specified as a list of start and end times in the format 'YYYY-MM-DD HH:MM:SS'.")

    # Check if trange is either a list or an array of length 2
    if not isinstance(trange, (list, np.ndarray)) or len(trange) != 2:
        raise ValueError(
            "trange must be specified as a list or array of length 2 in the format 'YYYY-MM-DD HH:MM:SS.")

    # Download the OMNI data (default level of 'hro_1min') for the specified timerange.
    omni_varnames = ['BX_GSE', 'BY_GSM', 'BZ_GSM', 'proton_density', 'Vx', 'Vy', 'Vz', 'SYM_H']
    omni_vars = spd.omni.data(trange=trange, varnames=omni_varnames, level=omni_level,
                             time_clip=time_clip)

    omni_time = ptt.get_data(omni_vars[0])[0]

    omni_bx_gse = ptt.get_data(omni_vars[0])[1]
    omni_by_gsm = ptt.get_data(omni_vars[1])[1]
    omni_bz_gsm = ptt.get_data(omni_vars[2])[1]
    omni_np = ptt.get_data(omni_vars[3])[1]
    omni_vx = ptt.get_data(omni_vars[4])[1]
    omni_vy = ptt.get_data(omni_vars[5])[1]
    omni_vz = ptt.get_data(omni_vars[6])[1]
    omni_sym_h = ptt.get_data(omni_vars[7])[1]

    time_imf = np.nanmedian(omni_time)
    b_imf_x = np.nanmedian(omni_bx_gse)
    b_imf_y = np.nanmedian(omni_by_gsm)
    b_imf_z = np.nanmedian(omni_bz_gsm)

    if (b_imf_z > 15 or b_imf_z < -18):
        warnings.warn(
        f"The given parameters produced the z-component of IMF field (b_imf_z) {b_imf_z} nT,"
        f"which is out of range in which model is valid (-18 nT < b_imf_z < 15 nT)"
        )

    time_imf_hrf = datetime.datetime.utcfromtimestamp(time_imf)
    np_imf = np.nanmedian(omni_np)
    vx_imf = np.nanmedian(omni_vx)
    vy_imf = np.nanmedian(omni_vy)
    vz_imf = np.nanmedian(omni_vz)
    sym_h_imf = np.nanmedian(omni_sym_h)
    v_imf = [vx_imf, vy_imf, vz_imf]
    b_imf = [b_imf_x, b_imf_y, b_imf_z]
    imf_clock_angle = np.arctan2(b_imf[1], b_imf[2]) * 180 / np.pi
    if imf_clock_angle < 0:
        imf_clock_angle += 180
    print("IMF parameters found:")

    if (verbose):
        print(tabulate(
            [["Time of observation (UTC)", time_imf_hrf],
             ["IMF Magnetic field [GSM] (nT)", b_imf],
             ["IMF Proton density (1/cm^-3)", np_imf],
             ["IMF Plasma velocity (km/sec)", v_imf],
             ["IMF clock angle (degrees)", imf_clock_angle],
             ["IMF Sym H", sym_h_imf]],
            headers=["Parameter", "Value"], tablefmt="fancy_grid", floatfmt=".2f",
            numalign="center"))

    # Check if the values are finite, if not then assign a default value to each of them
    if ~(np.isfinite(np_imf)):
        np_imf = 5
    if ~(np.isfinite(vx_imf)):
        vx_imf = -500
    if ~(np.isfinite(vy_imf)):
        vy_imf = 0
    if ~(np.isfinite(vz_imf)):
        vz_imf = 0
    if ~(np.isfinite(sym_h_imf)):
        sym_h_imf = -1

    m_proton = 1.672e-27  # Mass of proton in SI unit

    rho = np_imf * m_proton * 1.15

    #  Solar wind ram pressure in nPa, including roughly 4% Helium++ contribution
    p_dyn = 1.6726e-6 * 1.15 * np_imf * (vx_imf**2 + vy_imf**2 + vz_imf**2)

    if (p_dyn > 8.5 or p_dyn < 0.5):
        warnings.warn(
            f"The given parameters produced a dynamic pressure of {p_dyn} nPa which is out of"
            f" range in which model is valid (0.5 nPa < p_dyn < 8.5 nPa)",
        )
    param = [p_dyn, sym_h_imf, b_imf_y, b_imf_z, 0, 0, 0, 0, 0, 0]

    # Compute the dipole tilt angle
    ps = gp.recalc(time_imf)

    # Make a dictionary of all the solar wind parameters
    sw_dict = {}
    sw_dict['time'] = time_imf
    sw_dict['b_imf'] = b_imf
    sw_dict['np_imf'] = np_imf
    sw_dict['rho'] = rho
    sw_dict['ps'] = ps
    sw_dict['p_dyn'] = p_dyn
    sw_dict['sym_h'] = sym_h_imf
    sw_dict['imf_clock_angle'] = imf_clock_angle
    sw_dict['param'] = param

    return sw_dict


def model_run(*args):
    """
    Returns the value of the magnetic field at a given point in the model grid using three different
    models
    """
    import geopack.geopack as gp

    j = args[0][0]
    k = args[0][1]
    y_max = args[0][2]
    z_max = args[0][3]
    dr = args[0][4]
    m_p = args[0][5]
    ro = args[0][6]
    alpha = args[0][7]
    rmp = args[0][8]
    sw_params = args[0][9]
    model_type = args[0][-1]

    y0 = int(j * dr) - y_max
    z0 = int(k * dr) - z_max
    rp = np.sqrt(y0**2 + z0**2)  # Projection of r into yz-plane

    d_theta = np.pi/100

    for index in range(0, 100):

        theta = index * d_theta
        r = ro * (2/(1 + np.cos(theta))) ** alpha
        zp = r * np.sin(theta)  # not really in z direction, but a distance in yz plane
        x0 = r * np.cos(theta)

        if x0 == 0:
            signx = 1.0
        else:
            signx = np.sign(x0)

        if y0 == 0:
            signy = 1.0
        else:
            signy = np.sign(y0)

        if z0 == 0:
            signz = 1.0
        else:
            signz = np.sign(z0)

        if (rp <= zp):
            # print(index, rp, zp)
            # print(f'Value of theta = {theta}')

            y_coord = y0
            z_coord = z0
            x_shu = (r - m_p) * np.cos(theta)
            phi = np.arctan2(z0, y0)
            # print( j, k, theta, x_shu[j,k])

            if (abs(y0) == 0 or abs(z0) == 0):
                if(abs(y0) == 0):
                    y_shu = 0
                    z_shu = (r - m_p) * np.sin(theta)
                elif (abs(z0) == 0):
                    z_shu = 0
                    y_shu = (r - m_p) * np.sin(theta)
            else:
                z_shu = np.sqrt((rp - 1.0)**2/(1 + np.tan(phi)**(-2)))
                y_shu = z_shu/np.tan(phi)

            rho_sh = sw_params['rho'] * (1.509 * np.exp(x_shu/rmp) + .1285)

            m_proton = 1.672e-27  # Mass of proton in SI unit
            n_sh = rho_sh/m_proton

            y_shu = abs(y_shu)*signy
            z_shu = abs(z_shu)*signz

            # Cooling JGR 2001 Model, equation 9 to 12
            # the distance from the focus to the magnetopause surface
            A = 2
            ll = 3 * rmp/2 - x0
            b_msx = - A * (- sw_params['b_imf'][0] * (1 - rmp / (2 * ll)) + sw_params['b_imf'][1]
                        * (y0 / ll) + sw_params['b_imf'][2] * (z0 / ll))
            b_msy = A * (- sw_params['b_imf'][0] * (y0 / (2 * ll)) + sw_params['b_imf'][1]
                      * (2 - y0**2/( ll * rmp)) - sw_params['b_imf'][2] * (y0 * z0 / (ll * rmp)))
            b_msz = A * (- sw_params['b_imf'][0] * (z0 / (2 * ll)) - sw_params['b_imf'][1]
                      * (y0 * z0 / (ll * rmp)) + sw_params['b_imf'][2] * (2 - z0**2 / (ll * rmp)))
            try:
                if model_type == 't96':
                    bx_ext, by_ext, bz_ext = gp.t96.t96(sw_params['param'], sw_params['ps'], x_shu,
                                                                                       y_shu, z_shu)
                elif model_type == 't01':
                    bx_ext, by_ext, bz_ext = gp.t01.t01(sw_params['param'], sw_params['ps'], x_shu,
                                                                                       y_shu, z_shu)
            except:
                    print(f'Skipped for {x_shu, y_shu, z_shu}')
                    pass

            bx_igrf, by_igrf, bz_igrf = gp.igrf_gsm(x_shu, y_shu, z_shu)

            #print(j, k, bx_ext, bx_igrf)
            bx = bx_ext + bx_igrf
            by = by_ext + by_igrf
            bz = bz_ext + bz_igrf

            #if (np.sqrt(y_shu**2 + z_shu**2) > 31):
            #    shear = np.nan
            #    rx_en = np.nan
            #    va_cs = np.nan
            #    bisec_msp = np.nan
            #    bisec_msh = np.nan
            #else:
            shear = get_shear([bx, by, bz], [b_msx, b_msy, b_msz], angle_units="degrees")

            break

    return j, k, bx, by, bz, shear, x_shu, y_shu, z_shu, b_msx, b_msy, b_msz


def plot_shear(
    shear=None,
    sw_params=None,
    time_observation=None,
    angle_units="degrees",
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    model_type=None,
    dr=None,
    figure_file=None,
    figure_format="png",
    dpi=300,
    figure_size=(8, 6),
    data_file=None,
    save_figure=True
    ):

    fig, axs = plt.subplots(1, 1, figsize=figure_size)

    image_rotated = np.transpose(shear)
    # Smoothen the image
    image_smooth = sp.ndimage.filters.gaussian_filter(image_rotated, sigma=[5, 5], mode='nearest')
    im = axs.imshow(image_smooth, extent=[y_min, y_max, z_min, z_max], origin='lower',
                    cmap=plt.cm.viridis)
    divider = make_axes_locatable(axs)

    patch = patches.Circle((0, 0), radius=15, transform=axs.transData, fc='none', ec='k', lw=0.1)
    axs.add_patch(patch)
    #im.set_clip_path(patch)

    axs.tick_params(axis="both", direction="in", top=True, labeltop=False, bottom=True,
                    labelbottom=True, left=True, labelleft=True, right=True, labelright=False,
                    labelsize=14)

    axs.set_xlabel(r'Y [GSM, $R_\oplus$]', fontsize=18)
    axs.set_ylabel(r'Z [GSM, $R_\oplus$]', fontsize=18)

    cax = divider.append_axes("top", size="5%", pad=0.01)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal', ticks=None, fraction=0.05,
                        pad=0.01)
    cbar.ax.tick_params(axis="x", direction="in", top=True, labeltop=True, bottom=True,
                        labelbottom=False, pad=0.01, labelsize=14)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel(f'Shear Angle ({angle_units})', fontsize=18)

    # Write the timme range on the plot

    clock_angle_degrees = np.round(sw_params['imf_clock_angle'], 2)
    dipole_angle_degrees = np.round(sw_params['ps'] * 180 / np.pi, 2)

    axs.text(1.0, 0.5, f'Time of observation: {time_observation}', horizontalalignment='left',
             verticalalignment='center', transform=axs.transAxes, rotation=270, color='r')

    axs.text(0.01, 0.99, f'Clock Angle: {clock_angle_degrees}$^\circ$',
             horizontalalignment='left', verticalalignment='top', transform=axs.transAxes,
             rotation=0, color='r')

    axs.text(0.99, 0.99, f'Dipole tilt: {dipole_angle_degrees}$^\circ$',
             horizontalalignment='right', verticalalignment='top', transform=axs.transAxes,
             rotation=0, color='r')

    if (save_figure):
        fig.savefig(f'{figure_file}_{model_type}_{dr}dr.{figure_format}',
                    bbox_inches='tight', pad_inches=0.05, format=figure_format, dpi=dpi)
    plt.close()
    print(f"Figure saved to {figure_file}_{data_file}_{model_type}_{dr}dr.png")


def shear_angle_calculator(
    b_imf=None,
    np_imf=None,
    v_imf=None,
    dmp=0.5,
    dr=0.5,
    min_max_val = 15,
    y_min = None,
    y_max = None,
    z_min = None,
    z_max = None,
    model_type="t96",
    angle_units="radians",
    use_real_data=False,
    time_observation=None,
    dt=30,
    save_data=False,
    data_file="shear_data",
    plot_figure=False,
    save_figure=False,
    figure_file="shear_angle_calculator",
    figure_format="png",
    verbose=True,
):
    r"""
    Calculate the shear angle between the IMF and the Magnetosheath magnetic field. The code also
    saves the data to a hdf5 file and plots the figure.
    For this code to work, the following modules must be available/installed:
        - geopack (https://github.com/tsssss/geopack)
        - pyspedas (https://github.com/spedas/pyspedas)

    Parameters
    ----------
    b_imf : array of shape 1x3
        Interplanetary magnetic field vector. If not given, and "use_real_data" is set to False,
        then the code raises a ValueError. In order to use the real data, set "use_real_data" to
        True.
    np_imf : array of shape 1x1
        Interplanetary proton density. If not given, and "use_real_data" is set to False, then the
        code assumes a default value of 5.0 /cm^-3.
    v_imf : array of shape 1x3
        Interplanetary plasma bulk velocity vector. If not given, and "use_real_data" is set to
        False, then the code assumes a default value of [-500, 0, 0] km/sec.
    dmp : float
        Thickness of the magnetopause in earth radii units. Default is 0.5.
    dr : float
        Grid size in earth radii units. Default is 0.5.
    min_max_val : float
        The minimum and maximum values of the y and z-axis to be used for the computation. Default
        is 15, meaning the model will be computed for -15 < y < 15 and -15 < z < 15.
    y_min : float, optional
        The minimum value of the y-axis to be used for the computation. Default is None, in which
        case y_min is set to -min_max_val.
    y_max : float, optional
        The maximum value of the y-axis to be used for the computation. Default is None, in which
        case y_max is set to min_max_val.
    z_min : float, optional
        The minimum value of the z-axis to be used for the computation. Default is None, in which
        case z_min is set to -min_max_val.
    z_max : float, optional
        The maximum value of the z-axis to be used for the computation. Default is None, in which
        case z_max is set to min_max_val.
    model_type : str
        Model type to use. Default is "t96". Other option is "t01". Needs "geopack" to be installed.
    angle_units : str
        Units of the angle returned by the code. Default is "radians".
    use_real_data : bool
        If set to True, then the code will use the real data from the CDAWEB website. If set to
        False, then the code will use the default values for the parameters. For this to work, you
        must have the following modules installed:
        - pyspedas (https://github.com/spedas/pyspedas)
    time_observation : str
        Time of observation. If not given, then the code assumes the time of observation is the
        current time. The time of observation must be in the format "YYYY-MM-DDTHH:MM:SS".
    dt : float
        Time duration, in minutes, for which the IMF data is to be considered. Default is 30
        mcentered around the time of observation.
    save_data : bool
        If set to True, then the data will be saved to a hdf5 file. Default is False. Needs the h5py
        package to be installed.
    data_file : str
        Name of the hdf5 file to which data is to be saved. Default is "shear_data".
    plot_figure : bool
        If set to True, then the figure will be plotted and shown. Default is False.
    save_figure : bool
        If set to True, then the figure will be saved. Default is False.
    figure_file : str
        Name of the figure file. Default is "shear_angle_calculator".
    figure_format : str
        Format of the figure file. Default is "png". Other option can be "pdf".
    verbose : bool If set to True, then the code will print out the progress of the code at several
        points. Default is True. For this to work, you must have the following modules installed:
         - tabulate (https://pypi.org/project/tabulate/)
    """
    # Check if the required modules are installed
    try:
        import geopack.geopack as gp
    except ImportError:
        raise ImportError("geopack is not installed. Please install it using the command: pip install geopack or directly from the source at GitHub (https://github.com/tsssss/geopack).")

    if use_real_data:
        try:
            import pyspedas as spd
            import pytplot as ptt
        except ImportError:
            raise ImportError("pyspedas is not installed. Please install it using the command: pip install pyspedas")

    if save_data:
        try:
            import h5py as hf
        except ImportError:
            raise ImportError("h5py is not installed. Please install it using the command: pip install h5py")

    if verbose:
        try:
            from tabulate import tabulate
        except ImportError:
            raise ImportError("The required module 'tabulate' is not installed. Please install it using the command pip install tabulate.")

    if (b_imf is None and use_real_data is False):
        raise ValueError("Interplanetary magnetic field b_imf is not defined. If you do not wish to provide IMF magnetic field, set use_real_data to True")
    if use_real_data:
        if verbose:
            print("Attempting to download real data from the CDSAWEB website using PysSpedas \n")
        if time_observation is None:
            time_observation = "2020-02-16 17:00:00"
            time_observation = datetime.datetime.strptime(time_observation, "%Y-%m-%d %H:%M:%S")
            if verbose:
                print(f"Time of observation is not given. Defaulting to: {time_observation} UTC \n")
            if dt is None:
                print("dt, observation time range, is not defined. Setting dt to 30 minutes \n")
                dt = 30
            time_range = [(time_observation - datetime.timedelta(minutes=dt)).strftime(
                "%Y-%m-%d %H:%M:%S"), (time_observation + datetime.timedelta(minutes=dt)).strftime(
                "%Y-%m-%d %H:%M:%S")]
            if verbose:
                print(f"Downloading data from {time_range[0]} to {time_range[1]} \n")

        else:
            time_observation = datetime.datetime.strptime(time_observation, "%Y-%m-%d %H:%M:%S")
            if dt is None:
                print("dt, observation time range, is not defined. Setting dt to 30 minutes \n")
                dt = 30
            time_range = [(time_observation - datetime.timedelta(minutes=dt)).strftime(
                "%Y-%m-%d %H:%M:%S"), (time_observation + datetime.timedelta(minutes=dt)).strftime(
                "%Y-%m-%d %H:%M:%S")]
            if verbose:
                print(f"Downloading data from {time_range[0]} to {time_range[1]} \n")

        # Get the solar wind parameters for the model
        sw_params = get_sw_params(trange=time_range, omni_level="hro", verbose=verbose)

    if use_real_data is False:
        if verbose:
            print("Using default solar wind parameter values. If you want to use real time " +
                  "data, please use the function with the argument 'use_real_data=True' \n")
        v_imf = np.array([-500, 0.0, 0.0])
        np_imf = 5.0
        sym_h_imf = -30
        clock_angle = np.arctan2(b_imf[1], b_imf[2]) * 180 / np.pi

        m_proton = 1.672e-27  # Mass of proton in SI unit

        print(f"Hello, mass of proton={m_proton} kg and np_imf={np_imf} \n")
        rho = np_imf * m_proton * 1.15  # 1.15 instead of 1.0 to account for the alpha particles

        #  Solar wind ram pressure in nPa, including roughly 4% Helium++ contribution
        p_dyn = 1.6726e-6 * 1.15 * np_imf * np.linalg.norm(v_imf) ** 2

        # Compute the dipole tilt angle
        if use_real_data:
            time_dipole = calendar.timegm(time_observation.utctimetuple())
        else:
            time_dipole = calendar.timegm(datetime.datetime.utcnow().utctimetuple())
            #time_observation = datetime.datetime.strptime(time_observation, "%Y-%m-%d %H:%M:%S")
            #time_dipole = calendar.timegm(time_observation.utctimetuple())

        # Compute the dipole tilt angle
        dipole_tilt_angle = gp.recalc(time_dipole)

        sw_params = {}
        sw_params['b_imf'] = b_imf
        sw_params['v_imf'] = v_imf
        sw_params['rho'] = rho
        sw_params['p_dyn'] = p_dyn
        sw_params['sym_h'] = sym_h_imf
        sw_params['imf_clock_angle'] = clock_angle
        sw_params['ps'] = dipole_tilt_angle
        sw_params['param'] = [p_dyn, sym_h_imf, b_imf[1], b_imf[2], 0, 0, 0, 0, 0, 0]

        if verbose:
            print("Input parameters for the model:")
            print(tabulate(
                [["Solar wind dynamic pressure (nPa)", np.round(p_dyn, 3)],
                 ["IMF Sym H", np.round(sym_h_imf, 3)],
                 ["B_IMF_Y (nT)", np.round(b_imf[1], 3)],
                 ["B_IMF_Z (nT)", np.round(b_imf[2], 3)],
                 ["Clock Angle (degrees)", np.round(clock_angle, 3)]],
                headers=["Parameter", "Value"], tablefmt="fancy_grid", floatfmt=".2f",
                numalign="center"))

    if (sw_params['p_dyn'] > 8.5 or sw_params['p_dyn'] < 0.5):
        warnings.warn(
            f"The given parameters produced a dynamic pressure of {p_dyn} nPa which is out of"
            f" range in which model is valid (0.5 nPa < p_dyn < 8.5 nPa)",
        )

    if verbose:
        print("\n Computing Earth's magnetic field \n")

    # Set the min and max values of the x-axis and y-axis
    if y_min is None:
        y_min = -min_max_val
    if y_max is None:
        y_max = min_max_val
    if z_min is None:
        z_min = -min_max_val
    if z_max is None:
        z_max = min_max_val

    if dr is None:
        dr = 0.5
    if dmp is None:
        dmp = 0.5        

    n_arr_y = int((y_max - y_min) / dr) + 1
    n_arr_z = int((z_max - z_min) / dr) + 1

    bx = np.full((n_arr_y, n_arr_z), np.nan)
    by = np.full((n_arr_y, n_arr_z), np.nan)
    bz = np.full((n_arr_y, n_arr_z), np.nan)

    bx_ext = np.full((n_arr_y, n_arr_z), np.nan)
    by_ext = np.full((n_arr_y, n_arr_z), np.nan)
    bz_ext = np.full((n_arr_y, n_arr_z), np.nan)

    bx_igrf = np.full((n_arr_y, n_arr_z), np.nan)
    by_igrf = np.full((n_arr_y, n_arr_z), np.nan)
    bz_igrf = np.full((n_arr_y, n_arr_z), np.nan)

    b_msx = np.full((n_arr_y, n_arr_z), np.nan)
    b_msy = np.full((n_arr_y, n_arr_z), np.nan)
    b_msz = np.full((n_arr_y, n_arr_z), np.nan)

    x_shu = np.full((n_arr_y, n_arr_z), np.nan)
    y_shu = np.full((n_arr_y, n_arr_z), np.nan)
    z_shu = np.full((n_arr_y, n_arr_z), np.nan)

    rho_sh = np.full((n_arr_y, n_arr_z), np.nan)

    shear = np.full((n_arr_y, n_arr_z), np.nan)
    y_coord = np.full((n_arr_y, n_arr_z), np.nan)
    z_coord = np.full((n_arr_y, n_arr_z), np.nan)
    n_sh = np.full((n_arr_y, n_arr_z), np.nan)

    d_theta = np.pi / 100

    # Shue et al.,1998, equation 10
    ro = (10.22 + 1.29 * np.tanh(0.184 * (sw_params['b_imf'][2] + 8.14))) * (sw_params['p_dyn'])**(-1.0 / 6.6)

    # Shue et al.,1998, equation 11
    # alpha = (0.58 - 0.010 * b_imf_z) * (1 + 0.010 * p_dyn)
    alpha = (0.58 - 0.007 * sw_params['b_imf'][2]) * (1 + 0.024 * np.log(sw_params['p_dyn']))
    # Stand off position of the magnetopause
    rmp = ro * (2 / (1 + np.cos(0.0))) ** alpha

    len_y = int((y_max - y_min)/dr) + 1
    len_z = int((z_max - z_min)/dr) + 1

    p = mp.Pool()

    input = ((j, k, y_max, z_max, dr, dmp, ro, alpha, rmp, sw_params, model_type)
             for j in range(len_y) for k in range(len_z))

    print("Running the model \n")
    res = p.map(model_run, input)
    print("Model run complete \n")

    p.close()
    p.join()

    for r in res:
        j = r[0]
        k = r[1]
        bx[j, k] = r[2]
        by[j, k] = r[3]
        bz[j, k] = r[4]

        shear[j, k] = r[5]

        x_shu[j, k] = r[6]
        y_shu[j, k] = r[7]
        z_shu[j, k] = r[8]

        b_msx[j, k] = r[9]
        b_msy[j, k] = r[10]
        b_msz[j, k] = r[11]

    if (save_data):
        # Check if the data folder exists, if not then create it.
        if not os.path.exists("data_folder"):
            os.makedirs("data_folder")
            if verbose:
                print("Created data_folder folder")
        # Name of the file to save the data
        print(type(time_observation))
        tobs = str(time_observation).replace(" ", "_")
        fn = f'data_folder/{data_file}_{model_type}_{dr}dr_{tobs}.hdf5'
        # Save shear data to an hdf5 file
        try:
            with hf.File(fn, 'w') as f:
                f.create_dataset('x_shu', data=x_shu)
                f.create_dataset('y_shu', data=y_shu)
                f.create_dataset('z_shu', data=z_shu)
                f.create_dataset('bx_ext', data=bx_ext)
                f.create_dataset('by_ext', data=by_ext)
                f.create_dataset('bz_ext', data=bz_ext)
                f.create_dataset('bx_igrf', data=bx_igrf)
                f.create_dataset('by_igrf', data=by_igrf)
                f.create_dataset('bz_igrf', data=bz_igrf)
                f.create_dataset('bx', data=bx)
                f.create_dataset('by', data=by)
                f.create_dataset('bz', data=bz)
                f.create_dataset('shear', data=shear)
                f.create_dataset('n_sh', data=n_sh)
                f.create_dataset('rho_sh', data=rho_sh)
                f.create_dataset('b_msx', data=b_msx)
                f.create_dataset('b_msy', data=b_msy)
                f.create_dataset('b_msz', data=b_msz)
                f.close()
            print(f"Data saved to {data_file}_{model_type}_{dr}dr_{tobs}.hdf5")
        except Exception as e:
            print(e)
            print(
                f'Data not saved to file {fn}. Please make sure that file name is correctly assigned and that the directory exists and you have write permissions')

    if plot_figure:
        plot_shear(
            shear=shear,
            sw_params=sw_params,
            time_observation=time_observation,
            angle_units=angle_units,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            dr=dr, model_type=model_type,
            figure_file=figure_file,
            figure_format=figure_format,
            dpi=300,
            figure_size=(8,6),
            data_file=data_file,
            save_figure=save_figure
        )

    return shear