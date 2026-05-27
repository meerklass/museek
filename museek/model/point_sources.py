import numpy as np


def flux_HerculesA(freq_Hz):
    """Calculate HerculesA flux density. Accepts frequency in Hz."""
    freq_GHz = freq_Hz / 1e9  # Convert Hz to GHz
    F1410=47 #Jy
    F408=169 #Jy
    alpha=-np.log10(F1410/F408)/np.log10(1410/408.)
    return pow((freq_GHz/1.41),-alpha)*F1410

def flux_CenA(freq_Hz):
    """Calculate CenA flux density. Accepts frequency in Hz."""
    freq_GHz = freq_Hz / 1e9  # Convert Hz to GHz
    F1400=41 #Jy ###CHIPASS value, only this one is 1400 MHz rather than 1410 MHz (Parkes)
    F408=2746 #Jy
    alpha=-np.log10(F1400/F408)/np.log10(1400/408.)
    return pow((freq_GHz/1.40),-alpha)*F1400

def flux_3C161(freq_Hz):
    """Calculate 3C161 flux density. Accepts frequency in Hz."""
    freq_GHz = freq_Hz / 1e9  # Convert Hz to GHz
    F1410=18.5 #Jy
    F408=44.7 #Jy
    alpha=-np.log10(F1410/F408)/np.log10(1410/408.)
    return pow((freq_GHz/1.41),-alpha)*F1410

def flux_PKS1932_46(freq_Hz):
    """Calculate PKS1932-46 flux density. Accepts frequency in Hz."""
    freq_GHz = freq_Hz / 1e9  # Convert Hz to GHz
    F1410=12.6 #Jy
    F408=39.6 #Jy
    alpha=-np.log10(F1410/F408)/np.log10(1410/408.)
    return pow((freq_GHz/1.41),-alpha)*F1410

def flux_HydraA(freq_Hz):
    """Calculate HydraA flux density. Accepts frequency in Hz."""
    freq_GHz = freq_Hz / 1e9  # Convert Hz to GHz
    F1410=43.5 #Jy
    F408=132 #Jy
    alpha=-np.log10(F1410/F408)/np.log10(1410/408.)
    return pow((freq_GHz/1.41),-alpha)*F1410

def flux_3C353(freq_Hz):
    """Calculate 3C353 flux density. Accepts frequency in Hz."""
    freq_GHz = freq_Hz / 1e9  # Convert Hz to GHz
    F1410=54 #Jy
    F408=138 #Jy
    alpha=-np.log10(F1410/F408)/np.log10(1410/408.)
    return pow((freq_GHz/1.41),-alpha)*F1410

def flux_3C273(freq_Hz):
    """Calculate 3C273 flux density. Accepts frequency in Hz."""
    freq_GHz = freq_Hz / 1e9  # Convert Hz to GHz
    F1410=42 #Jy
    F408=55.10 #Jy
    alpha=-np.log10(F1410/F408)/np.log10(1410/408.)
    return pow((freq_GHz/1.41),-alpha)*F1410

def flux_3C237(freq_Hz):
    """Calculate 3C237 flux density. Accepts frequency in Hz."""
    freq_GHz = freq_Hz / 1e9  # Convert Hz to GHz
    F1410=6.6 #Jy
    F408=15.4 #Jy
    alpha=-np.log10(F1410/F408)/np.log10(1410/408.)
    return pow((freq_GHz/1.41),-alpha)*F1410

def flux_PictorA(freq_Hz):
    """Calculate PictorA flux density. Accepts frequency in Hz."""
    freq_GHz = freq_Hz / 1e9  # Convert Hz to GHz
    F1410=66. #Jy
    #alpha=0.85
    F408=166. #Jy
    alpha=-np.log10(F1410/F408)/np.log10(1410/408.)
    return pow((freq_GHz/1.41),-alpha)*F1410 #*0.8

def flux_1934_inter(freq_list_GHz):
    #408.MHz-8640.MHz
    a0,a1,a2,a3= -30.7667, 26.4908, -7.0977, 0.605334 #Lband-flux-calibrators.csv, katcofig

    #log(S) =a0+a1log(νG) +a2[log(νG)]2+a3[log(νG)]3+... #https://arxiv.org/pdf/1609.05940.pdf
    v=freq_list_GHz*1e3
    logS=a0+a1*np.log10(v)+a2*(np.log10(v))**2+a3*(np.log10(v))**3
    #print logS
    return 10**logS


def flux_1934_sd(freq_list_GHz):
    #PKS
    PKS_freq_GHz=[.408, 1.410, 2.700, 5.000, 8.400]
    PKS_flux=[6.24, 16.4, 11.5, 6.13, 3] #Jy
    p = np.polyfit(np.log10(PKS_freq_GHz),np.log10(PKS_flux),3)

    return 10**np.polyval(p,np.log10(freq_list_GHz))


def get_point_source_model(calibrator_name, freq_Hz):
    """
    Get point source flux and position for a given calibrator.

    :param calibrator_name: Name of the calibrator (e.g., 'HydraA', 'PictorA')
    :param freq_Hz: Frequency in Hz (can be scalar or array)
    :return: Tuple of (flux_Jy, ra_deg, dec_deg)
             - flux_Jy: Flux density in Janskys (same shape as freq_Hz)
             - ra_deg: Right ascension in degrees (scalar, placeholder=0 for now)
             - dec_deg: Declination in degrees (scalar, placeholder=0 for now)
    """
    # Mapping of calibrator names to functions
    models = {
        'HydraA': flux_HydraA,
        'PictorA': flux_PictorA,
        'CenA': flux_CenA,
        'HerculesA': flux_HerculesA,
        '3C161': flux_3C161,
        '3C353': flux_3C353,
        '3C273': flux_3C273,
        '3C237': flux_3C237,
        'PKS1932': flux_PKS1932_46,
        'PKS1934': flux_1934_sd,
    }

    # Calibrator positions (RA, Dec) in degrees, ICRS frame (J2000)
    # Coordinates from katcali/io.py (lines 286-306)
    positions = {
        'HydraA': (139.523617, -12.095502),          
        'PictorA': (79.9571708, -45.7788278),       
        'CenA': (201.365063, -43.019113),                         
        'HerculesA': (252.783945, 4.992588),                   
        '3C161': (96.792155, -5.884772),                       
        '3C353': (260.117381, -0.979642),            
        '3C273': (187.2779154, 2.0523883),          
        '3C237': (152.000125, 7.504541),             
        'PKS1932': (293.985833, -46.343611),
        'PKS1934': (294.854275, -63.712674)      
    }

    if calibrator_name not in models:
        raise ValueError(f"Unknown calibrator: {calibrator_name}. "
                        f"Available calibrators: {', '.join(models.keys())}")

    # Get flux using the appropriate function
    flux_func = models[calibrator_name]
    flux = flux_func(freq_Hz)

    # Get position (placeholder for now)
    ra, dec = positions.get(calibrator_name, (0.0, 0.0))

    return flux, ra, dec
