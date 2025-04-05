import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin

# System parameters
sample_rate = 10e9  # Sampling rate: 10 GHz
adc_bits = 7  # 7-bit ADC
full_range = 1.0  # Full-scale voltage: 1 V
levels = 2 ** adc_bits  # Number of quantization levels
step_size = full_range / levels  # Quantization step
freq_components = [0.2e9, 0.58e9, 1e9, 1.7e9, 2.4e9]  # Multi-tone frequencies

# Generate time array and composite signal
time = np.arange(0, 2 / min(freq_components), 1 / sample_rate)
input_waveform = sum(np.sin(2 * np.pi * freq * time) for freq in freq_components)

# Apply quantization
quantized_waveform = np.round(input_waveform / step_size) * step_size

# Calculate quantization error
error = input_waveform - quantized_waveform
error_variance = np.var(error)

# Theoretical uniform noise variance
theoretical_variance = (step_size ** 2) / 12

# Function to assess FIR filter impact on error
def evaluate_error_reduction(original_error, tap_count):
    # Create FIR filter coefficients
    filter_taps = firwin(tap_count, cutoff=0.5, window='hamming')
    
    # Filter the error signal
    filtered_output = lfilter(filter_taps, 1.0, original_error)
    
    # Compute corrected signal and its error
    adjusted_signal = original_error + filtered_output[:len(original_error)]
    new_error = original_error - adjusted_signal
    new_variance = np.var(new_error)
    
    return new_variance / theoretical_variance

# Analyze variance ratios for different tap counts
tap_range = range(2, 11)
ratio_values = [evaluate_error_reduction(error, taps) for taps in tap_range]

# Plot results
plt.figure()
plt.plot(tap_range, ratio_values, 'ro-', linewidth=1.5)
plt.title('Error Variance Ratio vs. FIR Tap Count (UIN_534001737)')
plt.xlabel('FIR Taps')
plt.ylabel('Ratio of Variances')
plt.grid(True)
plt.show()
