import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin

# System parameters
sample_freq = 10e9  # 10 GHz sampling rate
adc_resolution = 7  # 7-bit ADC
full_voltage = 1.0  # 1 V full range
lsb_value = full_voltage / (2 ** adc_resolution)  # Quantization step
amplitude = 0.5  # NRZ signal amplitude (0.5 V)
period = 1 / sample_freq  # Sampling interval

# Calculate required time constant
def compute_time_constant():
    tau_value = -period / np.log(lsb_value / amplitude)
    return tau_value

# Determine tau
time_constant = compute_time_constant()
print(f"Computed time constant: {time_constant:.2e} seconds")

# Time array
time_points = np.arange(0, 100 * period, period)

# Create NRZ waveform
nrz_waveform = np.random.choice([-amplitude, amplitude], size=len(time_points))

# Simulate two-channel time-interleaved ADC with mismatches
def simulate_ti_adc(input_signal, timing_error, offset_error, bw_variation):
    # Split into two channels
    chan1_data = input_signal[::2] + offset_error[0]
    chan2_data = input_signal[1::2] + offset_error[1]

    # Model bandwidth mismatch with low-pass filters
    filt1_coeffs = firwin(5, bw_variation[0])
    filt2_coeffs = firwin(5, bw_variation[1])
    chan1_data = lfilter(filt1_coeffs, 1.0, chan1_data)
    chan2_data = lfilter(filt2_coeffs, 1.0, chan2_data)

    # Apply timing skew
    shift_samples = int(timing_error * sample_freq)
    chan2_data = np.roll(chan2_data, shift_samples)

    # Merge channels
    output_signal = np.zeros(len(input_signal))
    output_signal[::2] = chan1_data
    output_signal[1::2] = chan2_data

    return output_signal

# Define mismatch characteristics
timing_skew = 1e-12  # 1 ps timing error
offset_deviation = [0.002, -0.002]  # ±2 mV offset
bw_difference = [0.45, 0.55]  # Cutoff frequency variation

# Generate TI-ADC output
ti_adc_result = simulate_ti_adc(nrz_waveform, timing_skew, offset_deviation, bw_difference)

# Calculate SNDR
input_power = np.var(nrz_waveform)
noise_power = np.var(nrz_waveform - ti_adc_result)
sndr_value = 10 * np.log10(input_power / noise_power)
print(f"Signal-to-Noise-and-Distortion Ratio: {sndr_value:.2f} dB")

# Visualize results
plt.figure(figsize=(10, 5))
plt.plot(time_points[:200], nrz_waveform[:200], 'b-', label='Input NRZ Waveform')
plt.plot(time_points[:200], ti_adc_result[:200], 'r--', label='TI-ADC Processed Output')
plt.title('Comparison of NRZ Input and TI-ADC Output')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (V)')
plt.legend(loc='best')
plt.grid(True)
plt.show()
