import numpy as np
import matplotlib.pyplot as plt

Nw = 100 # length of window

# Rectangular window
rect_window = np.ones(Nw)

# Hann window
hann_window = np.hanning(Nw)

# Gaussian window
sigma = 10 # standard deviation
x = np.linspace(0, Nw-1, Nw)
gaussian_window = np.exp(-(x - (Nw-1)/2)**2 / (2*sigma**2))

# Welch window
welch_window = np.hanning(Nw/2)
welch_window = np.concatenate((welch_window, welch_window[::-1]))

# Hamming window
hamming_window = np.hamming(Nw)

# Plot all windows
fig, axs = plt.subplots(1, 5, figsize=(15, 3), sharey=True)
fig.supxlabel("Sample")
fig.supylabel("Amplitude")

axs[0].plot(rect_window)
axs[0].set_title("(a) Rectangular")
axs[1].plot(hann_window)
axs[1].set_title("(b) Hann")
axs[2].plot(gaussian_window)
axs[2].set_title("(c) Gaussian")
axs[3].plot(welch_window)
axs[3].set_title("(d) Welch")
axs[4].plot(hamming_window)
axs[4].set_title("(e) Hamming")

for ax in axs:
    ax.set_ylim([0, 1.15])
    ax.set_xlim([0, Nw])
    ax.set_xticklabels(["0", "", "", "", "$N_{w-1}$"])

plt.show()
