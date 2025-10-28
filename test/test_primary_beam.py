"""
Test script for PrimaryBeam module.

This script tests and visualizes the MeerKAT UHF primary beam model:
- Beam patterns at multiple frequencies
- Beam area vs frequency
- Beam gain vs frequency at different angular offsets
- Both HH and VV polarizations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from museek.model.primary_beam import PrimaryBeam

# Configuration
BEAM_FILE = Path('/idia/projects/meerklass/beams/uhf/MeerKAT_U_band_primary_beam_aa_highres.npz')

# Test frequencies (MHz)
TEST_FREQS = [550, 700, 850, 1000]  # Span of UHF band

# Offsets to test (degrees from beam center)
TEST_OFFSETS = [0.0, 0.5, 1.0, 2.0, 3.0]


def plot_beam_patterns(beam: PrimaryBeam, polarization: str, figsize=(16, 10)):
    """Plot 2D beam patterns at different frequencies with log scale."""

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Create grid of (l, m) dimensionless direction cosines for plotting
    # Beam file extent is in pseudo-degrees (× 180/π), convert to dimensionless
    extent_rad = abs(beam.beam_extent_deg[0]) * np.pi / 180.0
    l = np.linspace(-extent_rad, extent_rad, 200)  # Dimensionless
    m = np.linspace(-extent_rad, extent_rad, 200)
    L, M = np.meshgrid(l, m)

    # Keep extent in degrees for plot axes
    extent_deg = abs(beam.beam_extent_deg[0])

    for idx, freq in enumerate(TEST_FREQS):
        ax = axes[idx]

        # Evaluate beam on grid
        freq_array = np.full_like(L, freq)

        # Evaluate beam: takes dimensionless direction cosines in (l, m) order
        beam_gain = beam.evaluate_beam(freq_array.flatten(),
                                        L.flatten(),
                                        M.flatten(),
                                        polarization)
        beam_gain = beam_gain.reshape(L.shape)

        # Plot with log scale to show sidelobes (convert axes to degrees for readability)
        l_deg = L * 180.0 / np.pi
        m_deg = M * 180.0 / np.pi

        # Use log scale with vmin at -40 dB (1e-4) to show sidelobes
        beam_gain_clipped = np.clip(beam_gain, 1e-4, 1.0)
        im = ax.imshow(beam_gain_clipped, extent=[-extent_deg, extent_deg, -extent_deg, extent_deg],
                      origin='lower', cmap='viridis', norm=LogNorm(vmin=1e-4, vmax=1))

        # Contours at dB levels: -30, -20, -10, -3 dB (must be increasing)
        contour_levels = 10**(np.array([-30, -20, -10, -3]) / 10)
        ax.contour(l_deg, m_deg, beam_gain, levels=contour_levels,
                  colors='white', linewidths=0.5, alpha=0.7)

        ax.set_xlabel('l (deg)', fontsize=11)
        ax.set_ylabel('m (deg)', fontsize=11)
        ax.set_title(f'{polarization} @ {freq} MHz')
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(im, ax=ax, label='Beam Gain')
        # Add dB labels to colorbar
        cbar.ax.set_ylabel('Beam Gain (log scale)', fontsize=10)

        # Add FWHM circle estimate
        # FWHM ≈ 1.2 * λ/D where D=13.5m for UHF
        wavelength = 3e8 / (freq * 1e6)  # meters
        fwhm_deg = np.degrees(1.2 * wavelength / 13.5)
        circle = plt.Circle((0, 0), fwhm_deg/2, fill=False,
                           color='red', linestyle='--', linewidth=2, label='Est. FWHM/2')
        ax.add_patch(circle)
        ax.legend(loc='upper right', fontsize=9)

    plt.suptitle(f'Primary Beam Patterns (Log Scale) - {polarization} Polarization', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'data/primary_beam_patterns_{polarization}_log.png', dpi=150, bbox_inches='tight')
    print(f"Saved: data/primary_beam_patterns_{polarization}_log.png")


def plot_beam_cuts(beam: PrimaryBeam, figsize=(16, 12)):
    """Plot 1D cuts through beam center along l and m axes."""

    fig, axes = plt.subplots(4, 2, figsize=figsize)

    # Create 1D cuts along l (m=0) and m (l=0)
    extent_rad = abs(beam.beam_extent_deg[0]) * np.pi / 180.0
    offsets = np.linspace(-extent_rad, extent_rad, 400)  # Dimensionless direction cosines
    offsets_deg = offsets * 180.0 / np.pi  # Convert to degrees for plotting

    for idx, freq in enumerate(TEST_FREQS):
        # Left column: HH polarization
        ax_hh = axes[idx, 0]
        # Right column: VV polarization
        ax_vv = axes[idx, 1]

        for pol, ax in [('HH', ax_hh), ('VV', ax_vv)]:
            # Cut along l axis (horizontal, m=0)
            freq_array_l = np.full_like(offsets, freq)
            m_zeros = np.zeros_like(offsets)
            beam_cut_l = beam.evaluate_beam(freq_array_l, offsets, m_zeros, pol)

            # Cut along m axis (vertical, l=0)
            l_zeros = np.zeros_like(offsets)
            beam_cut_m = beam.evaluate_beam(freq_array_l, l_zeros, offsets, pol)

            # Plot both cuts
            ax.semilogy(offsets_deg, beam_cut_l, 'b-', linewidth=2, label='Horizontal (l, m=0)')
            ax.semilogy(offsets_deg, beam_cut_m, 'r--', linewidth=2, label='Vertical (m, l=0)')

            # Add reference levels
            ax.axhline(1.0, color='k', linestyle=':', linewidth=1, alpha=0.5, label='Peak')
            ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='FWHM')
            ax.axhline(0.1, color='gray', linestyle=':', linewidth=0.8, alpha=0.3)
            ax.axhline(0.01, color='gray', linestyle=':', linewidth=0.8, alpha=0.3)

            ax.set_xlabel('Offset (deg)', fontsize=11)
            ax.set_ylabel('Beam Gain', fontsize=11)
            ax.set_title(f'{pol} @ {freq} MHz', fontsize=12)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_ylim(1e-4, 1.5)
            ax.set_xlim(-6, 6)
            if idx == 0:  # Only show legend on first row
                ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('1D Beam Cuts Through Center', fontsize=16)
    plt.tight_layout()
    plt.savefig('data/primary_beam_cuts.png', dpi=150, bbox_inches='tight')
    print("Saved: data/primary_beam_cuts.png")


def plot_beam_area_vs_frequency(beam: PrimaryBeam, figsize=(14, 10)):
    """Plot beam solid angle vs frequency with fractional deviation analysis."""

    # Get beam solid angles (already calculated during initialization)
    # These are at the frequencies in the beam file
    omega_HH = beam.beam_solid_angle_HH  # (n_freq,) in steradians
    omega_VV = beam.beam_solid_angle_VV

    # Get the corresponding frequencies
    freqs = beam._freq_MHz

    # Calculate theoretical beam area from effective aperture
    # For a circular dish: Ω = λ² / A_eff, where A_eff = η * π(D/2)²
    wavelength = 3e8 / (freqs * 1e6)  # meters (array)
    D = 13.5  # meters (MeerKAT UHF dish diameter)
    eta = 0.7  # Aperture efficiency (typical for dishes, adjust based on actual)
    A_physical = np.pi * (D / 2)**2  # Physical aperture area (scalar)
    A_eff_theory_constant = eta * A_physical  # Effective aperture area (scalar, frequency-independent)
    omega_theory = wavelength**2 / A_eff_theory_constant  # Beam solid angle (sr, array)

    # Calculate effective areas
    A_eff_HH = wavelength**2 / omega_HH  # Array: derived from beam model
    A_eff_VV = wavelength**2 / omega_VV  # Array: derived from beam model

    # Filter to 600-1000 MHz for fractional deviation plots
    freq_min, freq_max = 600.0, 1000.0
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_zoom = freqs[mask]
    omega_HH_zoom = omega_HH[mask]
    omega_VV_zoom = omega_VV[mask]
    A_eff_HH_zoom = A_eff_HH[mask]
    A_eff_VV_zoom = A_eff_VV[mask]

    # Calculate frequency-averaged values in the zoom range
    omega_HH_avg = np.mean(omega_HH_zoom)
    omega_VV_avg = np.mean(omega_VV_zoom)
    A_eff_HH_avg = np.mean(A_eff_HH_zoom)
    A_eff_VV_avg = np.mean(A_eff_VV_zoom)

    # Calculate fractional deviations
    omega_HH_frac = (omega_HH_zoom - omega_HH_avg) / omega_HH_avg * 100
    omega_VV_frac = (omega_VV_zoom - omega_VV_avg) / omega_VV_avg * 100
    A_eff_HH_frac = (A_eff_HH_zoom - A_eff_HH_avg) / A_eff_HH_avg * 100
    A_eff_VV_frac = (A_eff_VV_zoom - A_eff_VV_avg) / A_eff_VV_avg * 100

    # Plot: 2×2 layout
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # === Top-left: Beam solid angle (absolute) ===
    ax = axes[0, 0]
    ax.plot(freqs, omega_HH * 1e6, 'b-', linewidth=2, label='HH (beam model)')
    ax.plot(freqs, omega_VV * 1e6, 'r-', linewidth=2, label='VV (beam model)')
    ax.plot(freqs, omega_theory * 1e6, 'k--', linewidth=1.5, label=f'Theory (η={eta})')
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Beam Solid Angle (μsr)', fontsize=12)
    ax.set_title('Primary Beam Solid Angle', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === Top-right: Effective area (absolute) ===
    ax = axes[0, 1]
    ax.plot(freqs, A_eff_HH, 'b-', linewidth=2, label='HH (beam model)')
    ax.plot(freqs, A_eff_VV, 'r-', linewidth=2, label='VV (beam model)')
    ax.axhline(A_eff_theory_constant, color='k', linestyle='--', linewidth=1.5,
              label=f'Theory (η={eta})')
    ax.axhline(A_physical, color='gray', linestyle=':', linewidth=1.5,
              label=f'Physical Area (D={D}m)')
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Effective Area (m²)', fontsize=12)
    ax.set_title('Effective Aperture from Beam', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === Bottom-left: Beam solid angle fractional deviation ===
    ax = axes[1, 0]
    ax.plot(freqs_zoom, omega_HH_frac, 'b-', linewidth=1.5, label='HH')
    ax.plot(freqs_zoom, omega_VV_frac, 'r-', linewidth=1.5, label='VV')
    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Fractional Deviation from Mean (%)', fontsize=12)
    ax.set_title('Beam Solid Angle Ripples (600-1000 MHz)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(freq_min, freq_max)

    # === Bottom-right: Effective area fractional deviation ===
    ax = axes[1, 1]
    ax.plot(freqs_zoom, A_eff_HH_frac, 'b-', linewidth=1.5, label='HH')
    ax.plot(freqs_zoom, A_eff_VV_frac, 'r-', linewidth=1.5, label='VV')
    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Fractional Deviation from Mean (%)', fontsize=12)
    ax.set_title('Effective Area Ripples (600-1000 MHz)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(freq_min, freq_max)

    plt.tight_layout()
    plt.savefig('data/primary_beam_area_vs_frequency.png', dpi=150, bbox_inches='tight')
    print("Saved: data/primary_beam_area_vs_frequency.png")

    # Print statistics
    print(f"\nBeam Solid Angle Statistics:")
    print(f"  HH: {omega_HH.min()*1e6:.2f} - {omega_HH.max()*1e6:.2f} μsr")
    print(f"  VV: {omega_VV.min()*1e6:.2f} - {omega_VV.max()*1e6:.2f} μsr")
    print(f"\nEffective Area Statistics:")
    print(f"  HH: {A_eff_HH.min():.2f} - {A_eff_HH.max():.2f} m²")
    print(f"  VV: {A_eff_VV.min():.2f} - {A_eff_VV.max():.2f} m²")
    print(f"\nTheory (η={eta}):")
    print(f"  Beam area: {omega_theory.min()*1e6:.2f} - {omega_theory.max()*1e6:.2f} μsr")
    print(f"  A_eff (constant): {A_eff_theory_constant:.2f} m²")
    print(f"  Physical area: {A_physical:.2f} m²")


def _compare_beam_solid_angles(beam: PrimaryBeam, katbeam_model):
    """Compare beam solid angles from NPZ and katbeam models vs frequency."""

    # Use frequencies from the NPZ beam file
    freqs = beam._freq_MHz

    # Get NPZ beam solid angles (pre-calculated)
    omega_npz_HH = beam.beam_solid_angle_HH
    omega_npz_VV = beam.beam_solid_angle_VV

    # Calculate katbeam solid angles by numerical integration
    # Use same integration domain as NPZ beam
    extent_deg = 10.0  # Integration out to ±10 degrees (covers main lobe + near sidelobes)
    n_pts = 400  # Grid resolution for integration

    x_deg = np.linspace(-extent_deg, extent_deg, n_pts)
    y_deg = np.linspace(-extent_deg, extent_deg, n_pts)
    X_deg, Y_deg = np.meshgrid(x_deg, y_deg)

    # Differential solid angle element: dΩ = cos(θ) dx dy where θ is zenith angle
    # For small angles: cos(θ) ≈ 1 - (x² + y²)/2, but even simpler: dΩ ≈ dx dy (in radians²)
    # Since we're in degrees, need to convert: dΩ = (π/180)² dx dy
    dx_rad = np.radians(x_deg[1] - x_deg[0])
    dy_rad = np.radians(y_deg[1] - y_deg[0])
    dOmega = dx_rad * dy_rad  # Solid angle element in steradians

    omega_katbeam_HH = np.zeros_like(freqs)
    omega_katbeam_VV = np.zeros_like(freqs)

    print("\nCalculating katbeam solid angles (this may take a moment)...")
    for i, freq in enumerate(freqs):
        # Get katbeam power pattern (square of amplitude)
        beam_HH = np.abs(katbeam_model.HH(X_deg, Y_deg, freq))**2
        beam_VV = np.abs(katbeam_model.VV(X_deg, Y_deg, freq))**2

        # Integrate: Ω = ∫∫ P(x,y) dΩ where P is normalized beam power
        omega_katbeam_HH[i] = np.sum(beam_HH) * dOmega
        omega_katbeam_VV[i] = np.sum(beam_VV) * dOmega

        if i % 50 == 0:  # Progress indicator
            print(f"  Processed {i+1}/{len(freqs)} frequencies...")

    print("Done calculating katbeam solid angles.")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Beam solid angles
    ax = axes[0]
    ax.plot(freqs, omega_npz_HH * 1e6, 'b-', linewidth=2, label='NPZ HH')
    ax.plot(freqs, omega_npz_VV * 1e6, 'r-', linewidth=2, label='NPZ VV')
    ax.plot(freqs, omega_katbeam_HH * 1e6, 'b--', linewidth=2, label='katbeam HH')
    ax.plot(freqs, omega_katbeam_VV * 1e6, 'r--', linewidth=2, label='katbeam VV')
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Beam Solid Angle (μsr)', fontsize=12)
    ax.set_title('Beam Solid Angle Comparison', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Right: Ratio NPZ/katbeam
    ax = axes[1]
    ratio_HH = omega_npz_HH / omega_katbeam_HH
    ratio_VV = omega_npz_VV / omega_katbeam_VV
    ax.plot(freqs, ratio_HH, 'b-', linewidth=2, label='HH')
    ax.plot(freqs, ratio_VV, 'r-', linewidth=2, label='VV')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Beam Solid Angle Ratio (NPZ/katbeam)', fontsize=12)
    ax.set_title('Solid Angle Ratio', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/beam_solid_angle_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: data/beam_solid_angle_comparison.png")

    # Print statistics
    print("\nBeam Solid Angle Comparison:")
    print(f"  NPZ HH:     {omega_npz_HH.min()*1e6:.2f} - {omega_npz_HH.max()*1e6:.2f} μsr")
    print(f"  katbeam HH: {omega_katbeam_HH.min()*1e6:.2f} - {omega_katbeam_HH.max()*1e6:.2f} μsr")
    print(f"  Ratio HH:   {ratio_HH.min():.3f} - {ratio_HH.max():.3f} (mean: {ratio_HH.mean():.3f})")
    print(f"\n  NPZ VV:     {omega_npz_VV.min()*1e6:.2f} - {omega_npz_VV.max()*1e6:.2f} μsr")
    print(f"  katbeam VV: {omega_katbeam_VV.min()*1e6:.2f} - {omega_katbeam_VV.max()*1e6:.2f} μsr")
    print(f"  Ratio VV:   {ratio_VV.min():.3f} - {ratio_VV.max():.3f} (mean: {ratio_VV.mean():.3f})")

    # Calculate effective areas for reference
    wavelength = 3e8 / (freqs * 1e6)
    A_eff_npz_HH = wavelength**2 / omega_npz_HH
    A_eff_katbeam_HH = wavelength**2 / omega_katbeam_HH

    print(f"\nEffective Area Comparison (HH):")
    print(f"  NPZ:     {A_eff_npz_HH.min():.2f} - {A_eff_npz_HH.max():.2f} m²")
    print(f"  katbeam: {A_eff_katbeam_HH.min():.2f} - {A_eff_katbeam_HH.max():.2f} m²")


def compare_with_katbeam(beam: PrimaryBeam, figsize=(18, 12)):
    """Compare NPZ beam model with katbeam analytical model."""

    try:
        from katbeam import JimBeam
    except ImportError:
        print("katbeam not installed, skipping comparison")
        return

    # Create katbeam model
    katbeam_model = JimBeam('MKAT-AA-UHF-JIM-2020')

    # Test frequencies
    test_freqs = [600, 700, 850, 1000]

    # First, compare beam solid angles across frequency range
    _compare_beam_solid_angles(beam, katbeam_model)

    fig, axes = plt.subplots(len(test_freqs), 4, figsize=figsize)

    for row, freq in enumerate(test_freqs):
        # Create common grid in degrees
        extent_deg = 5.0  # ±5 degrees
        x_deg = np.linspace(-extent_deg, extent_deg, 200)
        y_deg = np.linspace(-extent_deg, extent_deg, 200)
        X_deg, Y_deg = np.meshgrid(x_deg, y_deg)

        # === Column 1: NPZ Beam (HH) ===
        # Convert degrees to direction cosines for NPZ beam
        l = X_deg.flatten() * np.pi / 180.0  # Small angle approx
        m = Y_deg.flatten() * np.pi / 180.0
        freq_array = np.full_like(l, freq)
        beam_npz_HH = beam.evaluate_beam(freq_array, l, m, 'HH').reshape(X_deg.shape)

        ax = axes[row, 0]
        im = ax.imshow(beam_npz_HH, extent=[-extent_deg, extent_deg, -extent_deg, extent_deg],
                      origin='lower', cmap='viridis', norm=LogNorm(vmin=1e-4, vmax=1))
        ax.set_title(f'NPZ Beam HH @ {freq} MHz', fontsize=11)
        ax.set_xlabel('x (deg)', fontsize=10)
        ax.set_ylabel('y (deg)', fontsize=10)
        plt.colorbar(im, ax=ax)

        # === Column 2: katbeam (HH) ===
        # katbeam returns AMPLITUDE (voltage pattern), need to square for power
        beam_katbeam_HH_amplitude = katbeam_model.HH(X_deg, Y_deg, freq)
        beam_katbeam_HH = np.abs(beam_katbeam_HH_amplitude)**2

        ax = axes[row, 1]
        im = ax.imshow(beam_katbeam_HH, extent=[-extent_deg, extent_deg, -extent_deg, extent_deg],
                      origin='lower', cmap='viridis', norm=LogNorm(vmin=1e-4, vmax=1))
        ax.set_title(f'katbeam HH @ {freq} MHz', fontsize=11)
        ax.set_xlabel('x (deg)', fontsize=10)
        ax.set_ylabel('y (deg)', fontsize=10)
        plt.colorbar(im, ax=ax)

        # === Column 3: Difference (HH) ===
        diff_HH = beam_npz_HH - beam_katbeam_HH

        ax = axes[row, 2]
        im = ax.imshow(diff_HH, extent=[-extent_deg, extent_deg, -extent_deg, extent_deg],
                      origin='lower', cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        ax.set_title(f'Difference @ {freq} MHz', fontsize=11)
        ax.set_xlabel('x (deg)', fontsize=10)
        ax.set_ylabel('y (deg)', fontsize=10)
        plt.colorbar(im, ax=ax, label='NPZ - katbeam')

        # === Column 4: 1D Cuts Comparison ===
        ax = axes[row, 3]

        # Horizontal cut (y=0)
        mid_idx = len(y_deg) // 2
        ax.semilogy(x_deg, beam_npz_HH[mid_idx, :], 'b-', linewidth=2, label='NPZ HH')
        ax.semilogy(x_deg, beam_katbeam_HH[mid_idx, :], 'r--', linewidth=2, label='katbeam HH')

        ax.set_xlabel('x offset (deg)', fontsize=10)
        ax.set_ylabel('Beam Gain', fontsize=10)
        ax.set_title(f'Horizontal Cut @ {freq} MHz', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(1e-4, 1.5)
        ax.set_xlim(-extent_deg, extent_deg)
        if row == 0:
            ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('NPZ Beam vs katbeam Analytical Model Comparison (HH Polarization)',
                 fontsize=16)
    plt.tight_layout()
    plt.savefig('data/beam_comparison_npz_vs_katbeam.png', dpi=150, bbox_inches='tight')
    print("Saved: data/beam_comparison_npz_vs_katbeam.png")

    # Print statistics
    print("\nComparison Statistics (HH Polarization):")
    for freq in test_freqs:
        # Recalculate for statistics
        l = X_deg.flatten() * np.pi / 180.0
        m = Y_deg.flatten() * np.pi / 180.0
        freq_array = np.full_like(l, freq)
        npz_vals = beam.evaluate_beam(freq_array, l, m, 'HH')
        # katbeam returns amplitude, square for power
        katbeam_vals = np.abs(katbeam_model.HH(X_deg.flatten(), Y_deg.flatten(), freq))**2

        diff = npz_vals - katbeam_vals
        rms_diff = np.sqrt(np.mean(diff**2))
        max_diff = np.abs(diff).max()

        print(f"  {freq} MHz: RMS diff = {rms_diff:.4f}, Max diff = {max_diff:.4f}")


def plot_beam_gain_vs_frequency(beam: PrimaryBeam, figsize=(14, 6)):
    """Plot fractional beam deviation vs frequency at different angular offsets.

    Plots (beam - beam_avg) / beam_avg to reveal ripple structure by removing
    the overall gain dependence on angular offset.
    """

    # Frequency range for plotting
    freq_min, freq_max = 600.0, 1000.0

    # High-resolution interpolated frequencies to show smooth curve
    freqs_hires = np.linspace(freq_min, freq_max, 4000)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for pol_idx, polarization in enumerate(['HH', 'VV']):
        ax = axes[pol_idx]

        for offset_deg in TEST_OFFSETS:
            # Calculate beam gain at offset from center
            # Assume offset is radial along l axis: l = offset, m = 0
            # Convert degrees to dimensionless direction cosine
            l_offset_hires = np.full_like(freqs_hires, np.sin(np.radians(offset_deg)))
            m_offset_hires = np.zeros_like(freqs_hires)
            beam_gain_hires = beam.evaluate_beam(freqs_hires, l_offset_hires, m_offset_hires,
                                                  polarization)

            # Calculate frequency-averaged beam gain for this offset
            beam_avg = np.mean(beam_gain_hires)

            # Calculate fractional deviation from mean
            fractional_dev = (beam_gain_hires - beam_avg) / beam_avg

            # Plot fractional deviation (as percentage)
            ax.plot(freqs_hires, fractional_dev * 100, linewidth=1.5,
                   label=f'Offset = {offset_deg}°')

        # Add zero line for reference
        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Frequency (MHz)', fontsize=12)
        ax.set_ylabel('Fractional Deviation from Mean (%)', fontsize=12)
        ax.set_title(f'{polarization} Polarization', fontsize=14)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(freq_min, freq_max)

    plt.suptitle('Beam Ripple Structure: Fractional Deviation vs Frequency (600-1000 MHz)', fontsize=16)
    plt.tight_layout()
    plt.savefig('data/primary_beam_gain_vs_frequency.png', dpi=150, bbox_inches='tight')
    print("Saved: data/primary_beam_gain_vs_frequency.png")


def test_beam_symmetry(beam: PrimaryBeam):
    """Test beam symmetry and compare HH vs VV."""

    freq = 850  # MHz (mid UHF band)

    # Test radial symmetry
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    radius_deg = 2.0  # degrees
    radius = np.sin(np.radians(radius_deg))  # Convert to dimensionless direction cosine

    print(f"\nBeam Symmetry Test at {freq} MHz, radius={radius_deg}°:")
    print(f"{'Angle (deg)':>12} {'HH Gain':>12} {'VV Gain':>12} {'HH/VV':>12}")
    print("-" * 50)

    gains_HH = []
    gains_VV = []

    for angle in angles:
        l = radius * np.cos(angle)  # Dimensionless
        m = radius * np.sin(angle)  # Dimensionless

        gain_HH = beam.evaluate_beam(freq, l, m, 'HH')
        gain_VV = beam.evaluate_beam(freq, l, m, 'VV')

        gains_HH.append(gain_HH)
        gains_VV.append(gain_VV)

        print(f"{np.degrees(angle):12.1f} {gain_HH:12.4f} {gain_VV:12.4f} {gain_HH/gain_VV:12.4f}")

    gains_HH = np.array(gains_HH)
    gains_VV = np.array(gains_VV)

    print(f"\nRadial symmetry (std/mean):")
    print(f"  HH: {gains_HH.std()/gains_HH.mean():.4f}")
    print(f"  VV: {gains_VV.std()/gains_VV.mean():.4f}")
    print(f"\nHH/VV ratio: {gains_HH.mean()/gains_VV.mean():.4f} ± {(gains_HH/gains_VV).std():.4f}")


def main():
    """Run all tests."""

    print(f"Loading beam from: {BEAM_FILE}")

    if not BEAM_FILE.exists():
        print(f"ERROR: Beam file not found: {BEAM_FILE}")
        print("Please ensure the beam file is in the correct location.")
        return

    # Load beam
    beam = PrimaryBeam(BEAM_FILE)

    print(f"\nBeam Model Info:")
    print(f"  Frequency range: {beam.freq_range_MHz[0]:.1f} - {beam.freq_range_MHz[1]:.1f} MHz")
    print(f"  Beam extent: ±{beam.beam_extent_deg[0]:.2f}° (l) × ±{beam.beam_extent_deg[1]:.2f}° (m)")
    print(f"  Polarizations: HH, VV")

    # Run tests
    print("\n" + "="*60)
    print("Generating beam pattern plots (log scale)...")
    plot_beam_patterns(beam, 'HH')
    plot_beam_patterns(beam, 'VV')

    print("\n" + "="*60)
    print("Generating 1D beam cuts...")
    plot_beam_cuts(beam)

    print("\n" + "="*60)
    print("Plotting beam area vs frequency...")
    plot_beam_area_vs_frequency(beam)

    print("\n" + "="*60)
    print("Plotting beam gain vs frequency...")
    plot_beam_gain_vs_frequency(beam)

    print("\n" + "="*60)
    print("Comparing with katbeam analytical model...")
    compare_with_katbeam(beam)

    print("\n" + "="*60)
    test_beam_symmetry(beam)

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("\nGenerated files:")
    print("  - data/primary_beam_patterns_HH_log.png")
    print("  - data/primary_beam_patterns_VV_log.png")
    print("  - data/primary_beam_cuts.png")
    print("  - data/primary_beam_area_vs_frequency.png")
    print("  - data/primary_beam_gain_vs_frequency.png")
    print("  - data/beam_solid_angle_comparison.png")
    print("  - data/beam_comparison_npz_vs_katbeam.png")


if __name__ == '__main__':
    main()
