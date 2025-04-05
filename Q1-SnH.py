#q1-lab4-p1-optional
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
sampling_freq = 10e9  # 10 GHz
signal_freq = 1e9     # 1 GHz
duration = 2e-9       # 2 ns

t = np.linspace(0, duration,200, endpoint=False )
# Generate the original sine wave
original_signal = np.sin(2 * np.pi * signal_freq * t)
# Sample the signal
sampled_indices = np.arange(0, len(t), int(sampling_freq / signal_freq))
print(sampled_indices)
sampled_signal = original_signal[sampled_indices]

# Generate hold signal
hold_signal = np.zeros_like(original_signal)
for i in range(len(sampled_indices) - 1):
    hold_signal[sampled_indices[i]:sampled_indices[i+1]] = sampled_signal[i]

plt.figure(figsize=(12, 8))
plt.plot(t, original_signal, label='Original Signal', color='pink')
plt.scatter(t[sampled_indices], sampled_signal, color='blue', label='Sampled Points')
plt.step(t, hold_signal, where='post', label='Hold Signal', color='red')

plt.title('Ideal Sample and Hold Output for Fin=1GHz, Fs=10GHz UIN534001737')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
