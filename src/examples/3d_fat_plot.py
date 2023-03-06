import numpy as np
import matplotlib.pyplot as plt

# Generate time array
t = np.linspace(0, 2*np.pi, 1000)

# Generate a noisy sine wave
x = np.sin(t) + 0.5*np.random.randn(len(t))

# Compute FFT of the signal
X = np.fft.fft(x)

# Compute frequency array
freq = np.fft.fftfreq(len(t), t[1] - t[0])

# Reconstruct each individual sine wave from the FFT
reconstructed_signal = np.zeros(len(t))
for n in range(len(X)):
    reconstructed_signal += (np.real(X[n]) * np.cos(2*np.pi*freq[n]*t) - np.imag(X[n]) * np.sin(2*np.pi*freq[n]*t)) / 1e3

# Plot the noisy signal and reconstructed signal
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t, x)
ax1.set_title('Noisy Signal')
ax2.plot(t, reconstructed_signal)
ax2.set_title('Reconstructed Signal')
plt.show()
