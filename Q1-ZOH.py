#q1-lab4-p2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Constants
F_INPUT = 1e9    # Input signal frequency: 1 GHz
F_SAMPLE = 10e9  # Sampling frequency: 10 GHz
R = 150           # Resistance in ohms
C = 0.05e-12        # Capacitance: 1 pF
AMPLITUDE = 1.0  # Input signal amplitude in volts
DURATION = 1 / F_INPUT  # Simulation duration: 1 sampling period (100 ps)

# Time parameters
Ts = 1 / F_SAMPLE  # Sampling period
time = np.arange(0, DURATION, Ts / 20)  # Simulation High-resolution time (10x oversampled)
sample_times = np.arange(0, DURATION, Ts)  # Discrete sampling points (2 points)

# Input signal: 1 GHz sine wave
input_signal = AMPLITUDE * np.sin(2 * np.pi * F_INPUT * time)

# Sampling clock: Square wave at 10 GHz (50% duty cycle)
clock = signal.square(2 * np.pi * F_SAMPLE * time, duty=0.5)
clock = (clock + 1) / 2  # Convert to 0/1

# Sample-and-hold simulation
snh_output = np.zeros_like(time)
last_held_value = 0  # Initial held value

# First, perform sample-and-hold on the input signal
for i in range(1, len(time)):
    if clock[i] > 0.5:  # Sample phase
        last_held_value = input_signal[i]
    #else:
    #    last_held_value = 0
    snh_output[i] = last_held_value

# Sample the S&H output at 10 GHz
sampled_snh = np.interp(sample_times, time, snh_output)

# RC circuit simulation on the sampled-and-held signal
tau = R * C  # Time constant
rc_output = np.zeros_like(time)
v_rc = 0  # Initial RC voltage

# Apply RC filtering to the S&H output
for i in range(1, len(time)):
    if clock[i] > 0.5:  # Sample phase
        dt = time[i] - time[i-1]
        v_rc = v_rc + (snh_output[i] - v_rc) * (1 - np.exp(-dt / tau))
    rc_output[i] = v_rc

# Sample the RC output at 10 GHz
sampled_rc = np.interp(sample_times, time, rc_output)

# Plotting all signals on one graph
plt.figure(figsize=(10, 6))
plt.plot(time * 1e12, input_signal,color='pink', label='1 GHz Input')
plt.plot(time * 1e12, clock, 'b--', label='10 GHz Sampling Clock')
plt.plot(time * 1e12, snh_output, 'g-', label='Voltage seen by charging circuit [Ideal]')
#plt.plot(sample_times * 1e12, sampled_snh, 'ro', label='S&H Sampled Points', markersize=8)
plt.plot(time * 1e12, rc_output, 'r-', label='RC Output [Non - Ideal]')
#plt.plot(sample_times * 1e12, sampled_rc, 'mo', label='RC Sampled Points', markersize=8)
plt.xlabel('Time (ps)')
plt.ylabel('Voltage (V)')
plt.title('Ideal and non-Ideal ZOH sampling Circuit (1 Period of 1GHz input at 10 GHz Sampling)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

# Print key metric
print(f"RC Time Constant: {tau*1e12:.3f} ps")
