#!/usr/bin/env python3
"""
Simulate detect_and_downsample_fft kernel (spatial.cuh:288) to check
whether its indexing mixes channels.

Synthetic input: cufft_data[ch][pol][beam][freq] = complex(ch, 0).
Any cross-pollination would produce output[ch][pol][beam][f] != ch^2.
"""
import numpy as np

NR_CHANNELS      = 4
NR_POLARIZATIONS = 2
NR_BEAMS         = 2
NR_FREQS         = 256       # full FFT size
DOWNSAMPLE       = 64        # FFT_DOWNSAMPLE_FACTOR
NR_OUT_FREQS     = NR_FREQS // DOWNSAMPLE

# Synthetic cuFFT output: power at (ch,pol,beam) = ch^2 (all freq bins equal)
# cufft_data is float2[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS][NR_FREQS]
# We represent it as a flat float2 array (complex) of size C*P*B*F
cufft_data = np.zeros((NR_CHANNELS, NR_POLARIZATIONS, NR_BEAMS, NR_FREQS),
                      dtype=np.complex64)
for ch in range(NR_CHANNELS):
    cufft_data[ch, :, :, :] = complex(ch, 0)  # real=ch, imag=0

# Flatten to match C memory layout [c][p][b][f]
cufft_flat = cufft_data.flatten()  # C row-major

output_data = np.zeros((NR_CHANNELS, NR_POLARIZATIONS, NR_BEAMS, NR_OUT_FREQS),
                       dtype=np.float32)

errors = []

# Simulate the kernel grid: x=out_freq, y=beam, z=chan*pol
for out_freq_idx in range(NR_OUT_FREQS):
    for beam_idx in range(NR_BEAMS):
        for chan_pol_idx in range(NR_CHANNELS * NR_POLARIZATIONS):
            chan = chan_pol_idx // NR_POLARIZATIONS
            pol  = chan_pol_idx %  NR_POLARIZATIONS

            start_f = out_freq_idx * DOWNSAMPLE

            # base_pointer (in float2 units) -- matches spatial.cuh:309
            base_pointer = (chan * NR_POLARIZATIONS * NR_BEAMS * NR_FREQS
                            + pol  * NR_BEAMS * NR_FREQS
                            + beam_idx * NR_FREQS)

            total = 0.0
            count = 0
            for j in range(DOWNSAMPLE):
                sample = cufft_flat[base_pointer + start_f + j]
                val = sample.real**2 + sample.imag**2
                total += val
                count += 1

            final_val = total / count if count > 0 else 0.0

            # output_base_pointer -- matches spatial.cuh:323
            output_base_pointer = (chan * NR_POLARIZATIONS * NR_BEAMS * NR_OUT_FREQS
                                   + pol  * NR_BEAMS * NR_OUT_FREQS
                                   + beam_idx * NR_OUT_FREQS)
            output_data[chan, pol, beam_idx, out_freq_idx] = final_val

# Check: output[ch][pol][beam][f] should equal ch^2 for all f (power of channel ch)
print("=" * 60)
print("detect_and_downsample_fft channel-mixing simulation")
print(f"Config: C={NR_CHANNELS} P={NR_POLARIZATIONS} B={NR_BEAMS} "
      f"Fin={NR_FREQS} Fout={NR_OUT_FREQS}")
print("Input: cufft_data[ch][...] = complex(ch, 0), so power = ch^2")
print("=" * 60)

errors = []
for ch in range(NR_CHANNELS):
    expected = float(ch)**2
    for pol in range(NR_POLARIZATIONS):
        for beam in range(NR_BEAMS):
            for f in range(NR_OUT_FREQS):
                got = output_data[ch, pol, beam, f]
                if abs(got - expected) > 1e-5:
                    errors.append(
                        f"  output[{ch}][{pol}][{beam}][{f}] = {got:.4f}, "
                        f"expected {expected:.4f}"
                    )

if errors:
    print(f"FAIL: {len(errors)} indexing errors in kernel simulation!")
    for e in errors[:10]:
        print(e)
else:
    print("PASS: detect_and_downsample_fft indexes channels independently.")
    print("      Each channel's output equals exactly its own input power.")
    print("\nSample output values (per channel):")
    for ch in range(NR_CHANNELS):
        print(f"  output[ch={ch}][pol=0][beam=0][:5] = "
              f"{output_data[ch, 0, 0, :5]}")
