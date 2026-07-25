"""
Engineering in Neuroscience - Assignment #1
EEG Signal Processing in a P300 Paradigm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, detrend

# **********************************************
# LOAD DATA
# **********************************************
mat = loadmat('Subject_A_Train.mat') #reads the binary .mat file into a Python dictionary
# getting these (epochs, samples, channels) from the dictionary mat
Signal        = mat['Signal'].astype(np.float64)
StimulusCode  = mat['StimulusCode'].astype(np.float64) #getting which row / col was flashing each moment, Values 1–6 = columns, 7–12 = rows, 0 = nothing flashing.
StimulusType  = mat['StimulusType'].astype(np.float64) #Gets whether each flash was a target (1) or non-target (0) flash.
#.astype(np.float64) converts from single precision (MATLAB default) to double precision so math works correctly.

fs = 240          # Sampling frequency (Hz) — stated in the docs
Cz_idx = 10       # Cz = channel 10 (0-indexed), label "Cz" = channel 11 in 1-indexed map

# **********************************************
# extract single-stimulus trials from epoch 0
# **********************************************
def extract_trials(epoch_signal, epoch_stim_code, samples_per_trial=None):
    """
    For a single character epoch, find each flash onset (StimulusCode goes non-zero)
    and grab `samples_per_trial` samples starting at that onset.
    Returns dict: {stim_code: list_of_signal_segments}
    """
    if samples_per_trial is None:
        samples_per_trial = int(0.667 * fs)  # ~667 ms window (long enough to cover P300 at ~300ms)

    trials = {i: [] for i in range(1, 13)} #Creates a dictionary with keys 1 through 12 (one for each possible row/column). Each key starts with an empty list that will be filled with signal segments.
    n = len(epoch_stim_code) # the total number of samples
    i = 0 # our position as we walk through the signal sample by sample.
    while i < n:
        code = int(epoch_stim_code[i])
        if code > 0: #At each sample, check if a flash is happening (intensified)
            end = min(i + samples_per_trial, n) # making sure we wont go past the end
            seg = epoch_signal[i:end] #Take a slice of 160 samples starting from the flash onset
            if len(seg) == samples_per_trial:
                trials[code].append(seg) #Only save the segment if it's exactly 160 samples long
            # skip to end of this flash
            while i < n and epoch_stim_code[i] > 0: # to avoid detecting the same flashing event twice
                i += 1
        else:
            i += 1 #If no flash is happening at this sample move to the next sample
    return trials

# **********************************************
# TASK 1 — First trial, Cz channel, time domain (raw)
# **********************************************
epoch0_cz   = Signal[0, :, Cz_idx]          # all time samples in epoch 0 (first character), channel Cz ---> gives 1D array of voltage values over time.
stim_code_0 = StimulusCode[0, :] #Gets the stimulus codes for that same first epoch, which row/column was flashing at each sample.

# Find first flash onset
first_flash_start = np.argmax(stim_code_0 > 0)
#stim_code_0 > 0 creates a True/False array, while np.argmax finds the index of the first True, the exact sample where the first flash begins.
trial_len = int(0.667 * fs)                 # 160 samples ≈ 667 ms
first_trial = epoch0_cz[first_flash_start : first_flash_start + trial_len] #Slices the signal from the first flash start to 160 samples later. This is our "first trial".
t_ms = np.arange(len(first_trial)) / fs * 1000  # time axis in ms

fig, ax = plt.subplots(figsize=(9, 4)) #creating the figure with axes.
ax.plot(t_ms, first_trial) #the time in ms is the x-axis while the EEG voltage is the Y=axis
ax.set_xlabel('Time (ms)') #labels
ax.set_ylabel('Amplitude (A/D units)')
ax.set_title('Task 1 — First trial from the Cz channel in the time domain, (raw data)')
ax.grid(True, alpha=0.3) # grids with 30% opacity
plt.tight_layout() #handeling the dimensions and labels
plt.savefig('task1_raw_time.png', dpi=150) #saving the figure
plt.show() #displaying the figure
print("Task 1 done.") #displaying a check in the compiler

# **********************************************
# TASK 2 — FFT of the raw first trial
# **********************************************
N   = len(first_trial) #160 (number of samples)
fft_vals = np.fft.fft(first_trial) # computes the Fast Fourier Transform, which converts the signal from time domain to frequency domain. Resulting in an array of 160 complex numbers.
freqs    = np.fft.fftfreq(N, d=1/fs) #Generates the frequency values (in Hz) that correspond to each position in fft_vals. d=1/fs tells it the time step between samples is 1/240 seconds.

# one-sided
#FFT output is symmetric so the second half is just a mirror of the first. So we only keep the positive frequencies using np.abs(). this gets the magnitude of each complex number (we only care about how strong each frequency is, not its phase or sign)
pos_mask = freqs >= 0
freqs_pos    = freqs[pos_mask]
fft_mag_pos  = np.abs(fft_vals[pos_mask])

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(freqs_pos, fft_mag_pos)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.set_title('Task 2 — FFT of First Trial vs frequency, (raw data)')
ax.set_xlim([0, fs/2]) #Sets x-axis from 0 to 120 Hz (the Nyquist frequency since we can't detect frequencies above half the sampling rate).
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task2_raw_fft.png', dpi=150)
plt.show()
print("Task 2 done.")

# **********************************************
# TASK 3 — Detrend + Bandpass filter (0.5–8 Hz, 4th order Butterworth)
#           then we have to repeat Tasks 1 & 2 on filtered signal
# **********************************************
def bandpass_filter(signal, lowcut=0.5, highcut=8.0, fs=240, order=4):
    #Defines the filter function. nyq = 120 is the Nyquist frequency (the maximum detectable frequency).
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    #Designs a 4th order Butterworth bandpass filter. The frequencies are normalized: 0.5/120 = 0.0042 and 8/120 = 0.067, so b and a are the filter coefficients.
    return filtfilt(b, a, signal) #Applies the filter forward then backward to avoid time-shifting the signal then Returns the filtered signal

# Detrend then filter the WHOLE Cz epoch (better filter behaviour than per-trial)
epoch0_cz_detrended = detrend(epoch0_cz)      # remove best-fit line as required by Fitting a straight line to the whole Cz epoch and subtracts it. Removes slow electrode drift.
epoch0_cz_filtered  = bandpass_filter(epoch0_cz_detrended)
#Passes the detrended signal through the Butterworth 0.5–8 Hz filter. Everything outside that range is removed.

first_trial_filt = epoch0_cz_filtered[first_flash_start : first_flash_start + trial_len]
#Cuts out the same 160-sample window from the filtered signal (same as Task 1 but now filtered).
# Task 3a — filtered time domain repeating task 1
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_ms, first_trial_filt)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude (A/D units)')
ax.set_title('Task 3 — First Trial, Cz Channel (Detrended + 0.5–8 Hz Filtered)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task3a_filtered_time.png', dpi=150)
plt.show()

# Task 3b — FFT of filtered signal, so basically repeating task 2
N_f        = len(first_trial_filt)
fft_filt   = np.fft.fft(first_trial_filt)
freqs_f    = np.fft.fftfreq(N_f, d=1/fs)
pos_f      = freqs_f >= 0
fft_filt_mag = np.abs(fft_filt[pos_f])

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(freqs_f[pos_f], fft_filt_mag)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.set_title('Task 3 — FFT of Filtered Signal')
ax.set_xlim([0, 20])   # zoom in — bandpass cuts above 8 Hz anyway
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task3b_filtered_fft.png', dpi=150)
plt.show()
print("Task 3 done.")

# **********************************************
# TASK 4 — Power spectrum (|FFT|²) of filtered signal
# **********************************************
power_filt = fft_filt_mag ** 2 #Squaring the FFT magnitude gives us the power at each frequency.

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(freqs_f[pos_f], power_filt)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (A/D units)²')
ax.set_title('Task 4 — Power Spectrum of Filtered Signal vs frequency')
ax.set_xlim([0, 20])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task4_power.png', dpi=150)
plt.show()
print("Task 4 done.")

# **********************************************
# TASK 5 — Log power vs. frequency
# **********************************************
log_power = 10 * np.log10(power_filt + 1e-12)
#Converts power to decibels (dB). The formula is 10 * log10(power). The + 1e-12 adds a very tiny number to prevent log10(0) which is negative infinity and would crash the program.
# dB measures the intensity of a signal or the reduction of noise
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(freqs_f[pos_f], log_power)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (dB)')
ax.set_title('Task 5 — Log Power Spectrum of Filtered Signal vs frequency')
ax.set_xlim([0, 20]) # setting the limits of the x-axis on the plot.
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task5_log_power.png', dpi=150)
plt.show()
print("Task 5 done.")

# **********************************************
# TASK 6 — Average responses: target vs. non-target
#           for the first character epoch (raw & filtered)
# **********************************************
stim_type_0 = StimulusType[0, :]   # 1 = target row/col, 0 = non-target

samples_per_trial = int(0.667 * fs)   # 160 samples

def get_target_nontarget_averages(signal_1d, stim_code_1d, stim_type_1d, spt): #Function that takes a signal and sorts every flash into "target" or "non-target" buckets, then averages each bucket.
    """Return mean target waveform and mean non-target waveform."""
    target_segs     = []
    nontarget_segs  = []
    #Two empty lists to collect segment
    n = len(stim_code_1d)
    i = 0
    while i < n:
        code = int(stim_code_1d[i])
        if code > 0:
            is_target = bool(stim_type_1d[i]) #Checks if the current flash is a target (StimulusType = 1) or not (0). bool() converts 1 --> True, 0 → False.
            end = min(i + spt, n)
            seg = signal_1d[i:end]
            if len(seg) == spt:
                if is_target: #Puts the 160-sample segment into the correct bucket either target or not.
                    target_segs.append(seg)
                else:
                    nontarget_segs.append(seg)
            while i < n and stim_code_1d[i] > 0:
                i += 1
        else:
            i += 1
    return np.mean(target_segs, axis=0), np.mean(nontarget_segs, axis=0)
#np.mean(..., axis=0) gets the average across all segments.

t_avg = np.arange(samples_per_trial) / fs * 1000

# RAW
avg_target_raw, avg_nontarget_raw = get_target_nontarget_averages(
    epoch0_cz, stim_code_0, stim_type_0, samples_per_trial)

# FILTERED
avg_target_filt, avg_nontarget_filt = get_target_nontarget_averages(
    epoch0_cz_filtered, stim_code_0, stim_type_0, samples_per_trial)

fig, axes = plt.subplots(1, 2, figsize=(14, 5)) # One window for both graphs

axes[0].plot(t_avg, avg_target_raw,    label='Target',     color='red')
axes[0].plot(t_avg, avg_nontarget_raw, label='Non-Target', color='blue', alpha=0.7)
axes[0].axvline(300, color='gray', linestyle='--', alpha=0.6, label='300 ms')  #Draws a vertical dashed line at 300ms here where we expect to see the P300 appear in the target response.
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude (A/D units)')
axes[0].set_title('Task 6 — Average Response (Raw)\nFirst Character Epoch, Cz')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_avg, avg_target_filt,    label='Target',     color='red')
axes[1].plot(t_avg, avg_nontarget_filt, label='Non-Target', color='blue', alpha=0.7)
axes[1].axvline(300, color='gray', linestyle='--', alpha=0.6, label='300 ms')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Amplitude (A/D units)')
axes[1].set_title('Task 6 — Average Response (Filtered)\nFirst Character Epoch, Cz')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task6_avg_responses.png', dpi=150)
plt.show()
print("Task 6 done.")
print("\nAll tasks complete! Figures saved as PNG files in the same directory.")
