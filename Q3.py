import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
sampling_freq = 10e9  # 10 GHz sampling frequency
sampling_period = 1 / sampling_freq  # Corresponding period
time_constant = 10e-12  # 10 ps time constant

# Create time array based on highest frequency (2.4 GHz)
time_step = sampling_period / 100
time_vector = np.arange(0, 1/(2.4e9), time_step)

# Signal frequency components
freq_list = [0.2e9, 0.58e9, 1e9, 1.7e9, 2.4e9]

# Generate composite multi-tone signal
composite_signal = sum(np.sin(2 * np.pi * freq * time_vector) for freq in freq_list)

# Model zero-order hold sampling behavior
def zoh_model(input_signal, samp_freq, tau, time):
    output_signal = np.zeros(len(input_signal))
    samples_per_period = int(1/(samp_freq * (time[1] - time[0])))
    
    for i in range(1, len(input_signal)):
        decay = np.exp(-(time[i] - time[i-1])/tau)
        if i % samples_per_period == 0:
            output_signal[i] = (output_signal[i-1] * decay + 
                              (1 - decay) * input_signal[i])
        else:
            output_signal[i] = output_signal[i-1] * decay
    return output_signal

# Apply sampling model to composite signal
processed_signal = zoh_model(composite_signal, sampling_freq, time_constant, time_vector)

# Visualize results
plt.figure(figsize=(12, 5))
plt.plot(time_vector, composite_signal, 'b-', label='Input Multi-Tone Waveform')
sample_indices = range(0, len(time_vector), int(1/(sampling_freq * time_step)))
plt.stem(time_vector[sample_indices], processed_signal[sample_indices], 
         linefmt='r-', markerfmt='go', label='Discrete Samples')
plt.title('Multi-Tone Signal Sampling Analysis (UIN534001737)')
plt.xlabel('Time (seconds)')
plt.ylabel('Signal Amplitude')
plt.grid(True)
plt.legend(loc='best')
plt.show()
