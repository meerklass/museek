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
from pathlib import Path
from museek.model.primary_beam import PrimaryBeam

# Configuration
BEAM_FILE = Path('data/MeerKAT_U_band_primary_beam_aa_highres.npz')

# Test frequencies (MHz)
TEST_FREQS = [550, 700, 850, 1000]  # Span of UHF band

# Offsets to test (degrees from beam center)
TEST_OFFSETS = [0.0, 0.5, 1.0, 2.0, 3.0]


def plot_beam_patterns(beam: PrimaryBeam, polarization: str, figsize=(16, 10)):
    """Plot 2D beam patterns at different frequencies."""

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

        # Plot (convert axes to degrees for readability)
        l_deg = L * 180.0 / np.pi
        m_deg = M * 180.0 / np.pi
        im = ax.imshow(beam_gain, extent=[-extent_deg, extent_deg, -extent_deg, extent_deg],
                      origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax.contour(l_deg, m_deg, beam_gain, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                  colors='white', linewidths=0.5, alpha=0.5)
        ax.set_xlabel('l (deg)', fontsize=11)
        ax.set_ylabel('m (deg)', fontsize=11)
        ax.set_title(f'{polarization} @ {freq} MHz')
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, label='Beam Gain')

        # Add FWHM circle estimate
        # FWHM ≈ 1.2 * λ/D where D=13.5m for UHF
        wavelength = 3e8 / (freq * 1e6)  # meters
        fwhm_deg = np.degrees(1.2 * wavelength / 13.5)
        circle = plt.Circle((0, 0), fwhm_deg/2, fill=False,
                           color='red', linestyle='--', linewidth=2, label='Est. FWHM/2')
        ax.add_patch(circle)
        ax.legend(loc='upper right', fontsize=9)

    plt.suptitle(f'Primary Beam Patterns - {polarization} Polarization', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'data/primary_beam_patterns_{polarization}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: data/primary_beam_patterns_{polarization}.png")


def plot_beam_area_vs_frequency(beam: PrimaryBeam, figsize=(12, 6)):
    """Calculate and plot beam solid angle vs frequency."""

    # Frequency array spanning UHF band
    freqs = np.linspace(beam.freq_range_MHz[0], beam.freq_range_MHz[1], 100)

    # Calculate beam area for both polarizations
    # Beam area (solid angle) = integral of beam pattern over sky
    # Simplified: use numerical integration on grid

    # Create grid in dimensionless direction cosines
    extent_rad = abs(beam.beam_extent_deg[0]) * np.pi / 180.0
    l = np.linspace(-extent_rad, extent_rad, 300)  # Dimensionless
    m = np.linspace(-extent_rad, extent_rad, 300)
    dl = l[1] - l[0]  # Dimensionless spacing
    dm = m[1] - m[0]
    L, M = np.meshgrid(l, m)

    omega_HH = []
    omega_VV = []

    print("Calculating beam areas...")
    for freq in freqs:
        freq_grid = np.full_like(L, freq)

        # HH beam area
        beam_HH = beam.evaluate_beam(freq_grid.flatten(),
                                      L.flatten(), M.flatten(), 'HH')
        beam_HH = beam_HH.reshape(L.shape)
        # Solid angle in steradians: Ω = ∫∫ P(l,m) dl dm
        # dl, dm are already dimensionless, so this is directly in steradians
        omega_HH.append(np.sum(beam_HH) * dl * dm)

        # VV beam area
        beam_VV = beam.evaluate_beam(freq_grid.flatten(),
                                      L.flatten(), M.flatten(), 'VV')
        beam_VV = beam_VV.reshape(L.shape)
        omega_VV.append(np.sum(beam_VV) * dl * dm)

    omega_HH = np.array(omega_HH)
    omega_VV = np.array(omega_VV)

    # Also calculate theoretical beam area: Ω = λ²/A_eff
    # For Gaussian beam: Ω = 2π(FWHM/2.355)² where FWHM ≈ 1.2λ/D
    wavelength = 3e8 / (freqs * 1e6)
    D = 13.5  # meters
    fwhm_rad = 1.2 * wavelength / D
    sigma_rad = fwhm_rad / 2.355
    omega_theory = 2 * np.pi * sigma_rad**2

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Beam area vs frequency
    ax = axes[0]
    ax.plot(freqs, omega_HH * 1e6, 'b-', linewidth=2, label='HH (measured)')
    ax.plot(freqs, omega_VV * 1e6, 'r-', linewidth=2, label='VV (measured)')
    ax.plot(freqs, omega_theory * 1e6, 'k--', linewidth=1.5, label='Theory (Gaussian)')
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Beam Area (μsr)', fontsize=12)
    ax.set_title('Primary Beam Solid Angle', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effective aperture: A_eff = λ²/Ω
    ax = axes[1]
    A_eff_HH = wavelength**2 / omega_HH
    A_eff_VV = wavelength**2 / omega_VV
    A_eff_theory = wavelength**2 / omega_theory

    ax.plot(freqs, A_eff_HH, 'b-', linewidth=2, label='HH')
    ax.plot(freqs, A_eff_VV, 'r-', linewidth=2, label='VV')
    ax.plot(freqs, A_eff_theory, 'k--', linewidth=1.5, label='Theory')
    ax.axhline(np.pi * (D/2)**2, color='gray', linestyle=':', linewidth=1.5,
              label=f'Physical Area ({D}m)')
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Effective Area (m²)', fontsize=12)
    ax.set_title('Effective Aperture from Beam', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/primary_beam_area_vs_frequency.png', dpi=150, bbox_inches='tight')
    print("Saved: data/primary_beam_area_vs_frequency.png")

    # Print statistics
    print(f"\nBeam Area Statistics:")
    print(f"HH: {omega_HH.min()*1e6:.2f} - {omega_HH.max()*1e6:.2f} μsr")
    print(f"VV: {omega_VV.min()*1e6:.2f} - {omega_VV.max()*1e6:.2f} μsr")
    print(f"A_eff HH: {A_eff_HH.min():.2f} - {A_eff_HH.max():.2f} m²")
    print(f"A_eff VV: {A_eff_VV.min():.2f} - {A_eff_VV.max():.2f} m²")


def plot_beam_gain_vs_frequency(beam: PrimaryBeam, figsize=(14, 6)):
    """Plot beam gain vs frequency at different angular offsets."""

    freqs = np.linspace(beam.freq_range_MHz[0], beam.freq_range_MHz[1], 200)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for pol_idx, polarization in enumerate(['HH', 'VV']):
        ax = axes[pol_idx]

        for offset_deg in TEST_OFFSETS:
            # Calculate beam gain at offset from center
            # Assume offset is radial along l axis: l = offset, m = 0
            # Convert degrees to dimensionless direction cosine
            l_offset = np.full_like(freqs, np.sin(np.radians(offset_deg)))
            m_offset = np.zeros_like(freqs)

            beam_gain = beam.evaluate_beam(freqs, l_offset, m_offset,
                                            polarization)

            ax.plot(freqs, beam_gain, linewidth=2,
                   label=f'Offset = {offset_deg}°')

        ax.set_xlabel('Frequency (MHz)', fontsize=12)
        ax.set_ylabel('Beam Gain', fontsize=12)
        ax.set_title(f'{polarization} Polarization', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Beam Gain vs Frequency at Different Offsets', fontsize=16)
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
    print("Generating beam pattern plots...")
    plot_beam_patterns(beam, 'HH')
    plot_beam_patterns(beam, 'VV')

    print("\n" + "="*60)
    print("Calculating beam area vs frequency...")
    plot_beam_area_vs_frequency(beam)

    print("\n" + "="*60)
    print("Plotting beam gain vs frequency...")
    plot_beam_gain_vs_frequency(beam)

    print("\n" + "="*60)
    test_beam_symmetry(beam)

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("\nGenerated files:")
    print("  - data/primary_beam_patterns_HH.png")
    print("  - data/primary_beam_patterns_VV.png")
    print("  - data/primary_beam_area_vs_frequency.png")
    print("  - data/primary_beam_gain_vs_frequency.png")


if __name__ == '__main__':
    main()
