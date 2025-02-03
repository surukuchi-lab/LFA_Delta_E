import sys

# Replace with the directory containing EggFileReader.py
module_dir = '/gpfs/gibbs/project/heeger/ek787/LFA_notebooks'
if module_dir not in sys.path:
    sys.path.append(module_dir)

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import interp1d
import scipy.stats
import json
import glob
import os
import re

from matplotlib.backends.backend_pdf import PdfPages

from EggFileReader import EggFileReader  # Reads in the simulation output

###############################################################################
#                   PART 0: HELPER FUNCTIONS TO BUILD PATHS
###############################################################################

def pitch_folder_format_snr(pitch):
    """
    For the SNR campaign, directories use:
       87._degrees   or   87.5_degrees
    """
    if abs(pitch - round(pitch)) < 1e-9:
        return f"{int(pitch)}._degrees"
    else:
        return f"{pitch}_degrees"

def get_radial_snr_basepath(pitch):
    """
    Returns the base directory for the radial SNR campaign.
    """
    folder = pitch_folder_format_snr(pitch)
    return f"/gpfs/gibbs/pi/heeger/ek787/LFA_simulations/20241209/Radial_SNR/{folder}/Trap_V00_00_05_nonCaterpillar/results"

def build_radial_snr_filename(pitch_folder_pitch, seed, pitch_val, radius, energy):
    """
    Returns the full path to the .egg file for the radial SNR campaign.
    """
    folder = pitch_folder_format_snr(pitch_folder_pitch)
    basepath = f"/gpfs/gibbs/pi/heeger/ek787/LFA_simulations/20241209/Radial_SNR/{folder}/Trap_V00_00_05_nonCaterpillar/results"
    return (f"{basepath}/"
            f"Seed{seed}_Angle{pitch_val:.9f}_Pos{radius:.7f}_Energy{energy:.9f}/"
            f"locust_mc_Seed{seed}_Angle{pitch_val:.9f}_Pos{radius:.7f}_Energy{energy:.9f}.egg")

def pitch_folder_format_llh(pitch, nominal_energy=18523.251):
    """
    For the LLH campaign, directories use:
       87.degrees_18523.251eV   or   87.5degrees_18523.251eV
    """
    if abs(pitch - round(pitch)) < 1e-9:
        return f"{int(pitch)}.degrees_{nominal_energy}eV"
    else:
        return f"{pitch}degrees_{nominal_energy}eV"

def get_llh_basepath(pitch, nominal_energy=18523.251):
    """
    Returns the base directory for the LLH campaign.
    """
    folder = pitch_folder_format_llh(pitch, nominal_energy)
    return f"/gpfs/gibbs/pi/heeger/ek787/LFA_simulations/20241209/DeltaE_grid_1/{folder}/Trap_V00_00_05_nonCaterpillar/results"

def build_llh_filename(pitch_folder_pitch, seed, pitch_val, radius, energy, nominal_energy=18523.251):
    """
    Returns the full path to the .egg file in the LLH campaign.
    """
    folder = pitch_folder_format_llh(pitch_folder_pitch, nominal_energy)
    basepath = f"/gpfs/gibbs/pi/heeger/ek787/LFA_simulations/20241209/DeltaE_grid_1/{folder}/Trap_V00_00_05_nonCaterpillar/results"
    return (f"{basepath}/"
            f"Seed{seed}_Angle{pitch_val:.9f}_Pos{radius:.7f}_Energy{energy:.9f}/"
            f"locust_mc_Seed{seed}_Angle{pitch_val:.9f}_Pos{radius:.7f}_Energy{energy:.9f}.egg")

def extract_sim_settings(basepath):
    """
    Extract (seed, pitch_val, radius, energy) from each directory name
    using the pattern:  'SeedNNN_AnglePPP.PosRRR_EnergyEEE'
    """
    sim_settings = []
    for path in glob.glob(basepath + "/*"):
        try:
            temp = re.findall("Seed(\d*)_Angle([\d\.]*)_Pos([-\d\.]*)_Energy([-\d\.]*)", path)[0]
            seed, pitch_val, radius, energy = int(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])
            sim_settings.append((seed, pitch_val, radius, energy))
        except:
            pass
    return sim_settings

###############################################################################
#                   PART 1: RADIAL SNR CALCULATION
###############################################################################

pitch_angles = [87.0, 87.5, 88.0, 88.5, 89.0, 89.5]

# Build a dictionary that captures the radial SNR info for each pitch
all_sim_settings = {}
for p in pitch_angles:
    bp = get_radial_snr_basepath(p)
    sim_settings = extract_sim_settings(bp)

    seed_set   = set(s[0] for s in sim_settings)
    pitch_set  = set(s[1] for s in sim_settings)
    radius_set = set(s[2] for s in sim_settings)
    energy_set = set(s[3] for s in sim_settings)

    all_sim_settings[p] = {
        'template_fn': build_radial_snr_filename,  # function to build .egg path
        'sim_settings': sim_settings,
        'seed_set':   seed_set,
        'pitch_set':  pitch_set,
        'radius_set': radius_set,
        'energy_set': energy_set
    }

#############################################
# Define a single global reference for noise
#############################################
reference_seed = 600
reference_seed_llh = 618
reference_pitch = 87.0
reference_radius = 0.16     # in meters
reference_energy = 18560.0  # eV (used in radial SNR campaign)
signal_len = 21000          # number of time samples used
noise_scaling = 13.5/(0.624*np.sqrt(1.88))
sampling_rate = 21e6

# Load a single reference file for noise
ref_file_name = all_sim_settings[reference_pitch]['template_fn'](
    reference_pitch, reference_seed, reference_pitch, reference_radius, reference_energy
)
file_for_noise = EggFileReader(ref_file_name)
ts_ref = file_for_noise.quick_load_ts_stream()
mc_truth_ref = ts_ref[0, 0][:signal_len]

# Global noise
sigma_global = np.max(np.abs(mc_truth_ref)) * noise_scaling
noise_global = np.squeeze(
    np.random.normal(loc=0, scale=sigma_global/2, size=(len(mc_truth_ref), 2)).view(np.complex128)
)

#############################################
# Compute the Radial SNR for each pitch
#############################################
pitch_snr_results = {}
for pitch in pitch_angles:
    snr_list = []
    sim_dict = all_sim_settings[pitch]
    radius_list = sorted(sim_dict['radius_set'])
    
    for r in radius_list:
        # Build filename
        fname = sim_dict['template_fn'](pitch, reference_seed, pitch, r, reference_energy)
        file_snr = EggFileReader(fname)
        ts_snr = file_snr.quick_load_ts_stream()
        mc_truth = ts_snr[0, 0][:signal_len]

        # Add same global noise
        signal = mc_truth + noise_global

        # FFT-based SNR
        yf_noise = fftshift(fft(noise_global[:signal_len])) / np.sqrt(signal_len)
        yf_electron = fftshift(fft(mc_truth[:signal_len])) / np.sqrt(signal_len)

        # SNR definition
        SNR = np.sum(np.abs(yf_electron)**2) / np.mean(np.abs(yf_noise)**2)
        snr_list.append(SNR)
    
    pitch_snr_results[pitch] = {
        'radii': radius_list,
        'snr': snr_list
    }

# Plot and save the radial SNR
plt.figure(figsize=(8,5))
for p in pitch_angles:
    radii = pitch_snr_results[p]['radii']
    snr   = pitch_snr_results[p]['snr']
    plt.plot(radii, snr, label=f"Pitch {p} deg")
plt.xlabel("Radius (m)")
plt.ylabel("SNR")
plt.title("Radial SNR for Different Pitch Angles")
plt.legend()
plt.grid(True)
plt.savefig("Radial_SNR_plot.pdf")  # Save radial SNR PDF
plt.show()

###############################################################################
#                   PART 2: LLH CALCULATION (pitch vs. radius)
###############################################################################

nominal_energies_for_llh = [18523.251]

def delta_LLH_level(sigma, ndof=2):
    """
    Convert a 'sigma' in 1D to a delta-LLH contour in 'ndof' dimensions
    using chi-square coverage. We use this for 1σ, 2σ, 3σ levels in 2D.
    """
    prob = scipy.stats.chi2(df=1).cdf(sigma**2)
    dchi2 = scipy.stats.chi2(df=ndof).isf(1 - prob)
    dLLH = dchi2 / 2.0
    return dLLH

# Build SNR interpolators for amplitude scaling
snr_interpolators = {}
for p in pitch_angles:
    rad_list = pitch_snr_results[p]['radii']
    snr_list = pitch_snr_results[p]['snr']
    snr_interpolators[p] = interp1d(rad_list, snr_list, kind='cubic', fill_value="extrapolate")

def load_waveform_for_LLH(campaign_info, pitch_dir, pitch_val, radius, energy, seed=reference_seed_llh):
    """
    Load the 'expectation' (simulated) waveform from the LLH directory scenario.
    We keep the same function signature but we'll fix the energy
    to the given 'energy' (nominal) and use the provided pitch, radius.
    """
    fname = campaign_info[pitch_dir]['template_fn'](pitch_dir, seed, pitch_val, radius, energy)
    f = EggFileReader(fname)
    ts = f.quick_load_ts_stream()
    return ts[0, 0][:signal_len]

def compute_measured_signal(p_truth, r_truth, energy_truth, campaign_info):
    """
    Build a 'measured' signal from the truth scenario:
      1) Load wave at (p_truth, reference_radius, energy_truth),
      2) Scale amplitude from reference_radius -> r_truth,
      3) Add the same global noise (noise_global).
    """
    truth_ref = load_waveform_for_LLH(campaign_info, p_truth, p_truth, reference_radius, energy_truth)
    snr_ref = snr_interpolators[p_truth](reference_radius)
    snr_target = snr_interpolators[p_truth](r_truth)
    scale_factor = np.sqrt(snr_target / snr_ref)
    truth_wave_scaled = truth_ref * scale_factor
    return truth_wave_scaled + noise_global

def compute_LLH_map(pitch_list, radius_list, true_radius, measured_signal, campaign_info, nominal_energy):
    """
    For each (pitch, r) in pitch_list x radius_list:
      1) Load wave at (pitch, reference_radius, nominal_energy)
      2) Scale amplitude to 'r'
      3) Compute Gaussian LLH vs. measured_signal
    Returns a 2D array LLH_map[pitch_index, radius_index].
    
    'true_radius' is included here because the function signature originally
    had a 'radius' argument. We'll keep it but only use it to keep consistent
    with your snippet structure. It's not used directly except for naming.
    """
    LLH_map = np.zeros((len(pitch_list), len(radius_list))) * np.nan

    for i, p in enumerate(pitch_list):
        # SNR at reference radius for this pitch
        snr_ref = snr_interpolators[p](reference_radius)

        for j, r_val in enumerate(radius_list):
            try:
                wave_ref = load_waveform_for_LLH(campaign_info, p, p, reference_radius, nominal_energy)
                snr_r = snr_interpolators[p](r_val)
                amp_factor = np.sqrt(snr_r / snr_ref)
                wave_scaled = wave_ref * amp_factor

                # Gaussian LLH
                LLH_val = 0.5 / sigma_global**2 * np.sum(np.abs(measured_signal - wave_scaled)**2)
                LLH_map[i, j] = LLH_val
            except:
                pass

    return LLH_map

###############################################################################
# Main analysis loop: vary "true pitch" and "true radius" for each nominal_energy
###############################################################################
analysis_results = {}

for nominal_energy in nominal_energies_for_llh:
    # Build campaign_info for the current nominal_energy
    campaign_info = {}
    for p in pitch_angles:
        bp = get_llh_basepath(p, nominal_energy=nominal_energy)
        settings_here = extract_sim_settings(bp)
        seed_set   = set(s[0] for s in settings_here)
        pitch_set  = set(s[1] for s in settings_here)
        radius_set = set(s[2] for s in settings_here)
        energy_set = set(s[3] for s in settings_here)

        campaign_info[p] = {
            'template_fn': lambda pitch_folder, seed, pitch_val, rad, E: build_llh_filename(
                                pitch_folder, seed, pitch_val, rad, E, nominal_energy=nominal_energy),
            'sim_settings': settings_here,
            'seed_set':   seed_set,
            'pitch_set':  pitch_set,
            'radius_set': radius_set,
            'energy_set': energy_set
        }

    # Find the common radius set across all pitches (instead of common_energy_set)
    common_radius_set = None
    for p in pitch_angles:
        r_set = campaign_info[p]['radius_set']
        if common_radius_set is None:
            common_radius_set = r_set
        else:
            common_radius_set = common_radius_set.intersection(r_set)

    if not common_radius_set:
        print(f"[WARNING] No common radii found for nominal_energy={nominal_energy} across all pitches!")
        continue

    # Sort them (this will be our "radius_list" in the LLH scan)
    common_radius_list = sorted(common_radius_set)

    # We'll define a grid of "true" radii to test (the MC truth radius)
    radii_grid = np.linspace(0.02, 0.30, 10)

    # We'll store all LLH maps, plus other possible resolution info
    LLH_map_storage = {}

    for true_pitch in pitch_angles:
        for true_radius in radii_grid:

            # Define MC truth scenario
            mc_pitch  = true_pitch
            mc_radius = true_radius
            mc_energy = nominal_energy

            # 1) Generate measured signal from (mc_pitch, mc_radius, mc_energy)
            measured_signal = compute_measured_signal(mc_pitch, mc_radius, mc_energy, campaign_info)
            
            # 2) Compute LLH map on the grid (pitch_angles x common_radius_list)
            pitch_set_sorted = sorted(pitch_angles)
            radius_set_sorted = sorted(common_radius_list)
            LLH_map = compute_LLH_map(
                pitch_set_sorted, radius_set_sorted, mc_radius, measured_signal, 
                campaign_info, nominal_energy
            )
            LLH_map_storage[(mc_pitch, round(mc_radius, 3))] = LLH_map

            # 3) Now replicate the exact plotting snippet (but now pitch vs. radius)
            LLH0 = np.min(LLH_map)
            dLLH = LLH_map - LLH0

            # Build the mesh for contouring
            P, R = np.meshgrid(np.array(pitch_set_sorted), np.array(radius_set_sorted), indexing='ij')

            fig = plt.figure(figsize=(8, 5))
            # Use imshow with extent; note we replaced energy with radius
            plt.imshow(dLLH,
                       extent=[min(pitch_set_sorted), max(pitch_set_sorted), 
                               min(radius_set_sorted), max(radius_set_sorted)],
                       origin="lower", aspect="auto")
            cb = plt.colorbar()
            cb.set_label("$\\Delta \\log\\mathcal{L}$")

            # Overlaid contours at 1σ, 2σ, 3σ (2D)
            levels = [delta_LLH_level(sigma=i, ndof=2) for i in [1,2,3]]
            cs = plt.contour(P, R, dLLH, levels=levels, colors=["white", "gray", "black"])

            plt.plot([], [], color="white", label="$1\\sigma\\,\\,(68.3\\%)$")
            plt.plot([], [], color="gray",  label="$2\\sigma\\,\\,(95.4\\%)$")
            plt.plot([], [], color="black", label="$3\\sigma\\,\\,(99.7\\%)$")

            # Mark MC truth (pitch, radius)
            plt.scatter(mc_pitch, mc_radius, color="r", label="MC truth")
            plt.legend(loc='lower right')

            # Compute the SNR at the truth point
            snr_val = snr_interpolators[mc_pitch](mc_radius)

            plt.xlabel("θ $(^0)$")
            plt.ylabel("Radius (m)")
            plt.title(f"LFA (V05) θ vs. $r$ LLH scan\n(SNR = {snr_val:.3f} for $r$ = {mc_radius:.2f} m)")

            plt.tight_layout()

            # Save figure
            png_name = (
                "./LLH_landscape_pitch%.9f_pitch_%.2f_radius_%.9f_energy_1.0_ms_"
                "pitch_radius_fine_scan_SNR_%.2f.png"
                % (mc_pitch, mc_pitch, mc_radius, snr_val)
            )
            pdf_name = (
                "./LLH_landscape_pitch%.9f_pitch_%.2f_radius_%.9f_energy_1.0_ms_"
                "pitch_radius_fine_scan_SNR_%.2f.pdf"
                % (mc_pitch, mc_pitch, mc_radius, snr_val)
            )

            plt.savefig(png_name, dpi=600, bbox_inches="tight")
            with PdfPages(pdf_name) as pdf:
                pdf.savefig(fig)

            # Print out the sigma range in pitch, radius from the first contour path
            print("Truth: Pitch=%.3f deg, Radius=%.3f m, Energy=%.3f eV" % (mc_pitch, mc_radius, mc_energy))
            
            for i in range(len(levels)):
                # If there is at least one path in this contour level
                if i < len(cs.collections):
                    ccoll = cs.collections[i]
                    if len(ccoll.get_paths()) > 0:
                        pth = ccoll.get_paths()[0]
                        v = pth.vertices
                        x = v[:,0]  # pitch
                        y = v[:,1]  # radius
                        pitch_half_range = (max(x) - min(x))/2
                        radius_half_range = (max(y) - min(y))/2
                        print("%d sigma: %.6f mdeg, %.5f m"
                              % ((i+1), 1000*pitch_half_range, radius_half_range))

            plt.close(fig)

    # Store results
    analysis_results[nominal_energy] = {
        'pitch_angles': pitch_angles,
        'radii_grid': radii_grid,
        'common_radius_list': common_radius_list,
        'LLH_map_storage': LLH_map_storage
    }

if nominal_energies_for_llh:
    e_last = nominal_energies_for_llh[-1]
    info = analysis_results[e_last]
    pitch_list = info['pitch_angles']
    r_grid = info['radii_grid']
    deltaE_map = np.zeros((len(pitch_list), len(r_grid)))
    analysis_results[e_last]['deltaE_map'] = deltaE_map

    plt.figure(figsize=(10,8))
    plt.imshow(deltaE_map,
               extent=[r_grid.min(), r_grid.max(), pitch_list[0], pitch_list[-1]],
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label=r"$\Delta E_{1\sigma}$ (eV)")
    plt.xlabel("Radius (m)")
    plt.ylabel("Pitch Angle (°)")
    plt.title(f"1σ Energy Resolution (ΔE) grid\n(Pitch Angle vs. Radius) for E = {e_last:.3f} eV")
    plt.savefig(f'Energy_resolution_map_v3_theta_vs_R_for_E_{e_last:.3f}_eV.pdf')
    plt.show()
