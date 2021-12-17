import numpy as np
import matplotlib.pyplot as plt
import tes_simulator as tessim
import noise_gen as noise
import pandas as pd
import os
import sys
import datetime
import scipy as sp
import warnings

warnings.filterwarnings('ignore')

# Parameters
p_opt_pW = 0.3061 # 100 ghz LFT
nep_aW_over_rthz = 9.778 # 100 ghz LFT
sampling_rate_hz = 10000
time_sec = 3600
fknee_hz = 0.1
alpha = 2
length = int(time_sec * sampling_rate_hz + 1)
sigma = nep_aW_over_rthz * np.sqrt(sampling_rate_hz) / 1e18

# Define TES
optical_loading_power = p_opt_pW * 1e-12
biasing_current = 25.e-6
tes_saturation_power = 2.5 * optical_loading_power
tes_leg_thermal_conductivity = tes_saturation_power/((1.71*0.1)**4.-0.1**4.)*4.*(1.71*0.1)**(4.-1.)
tes_heat_capacity = 33.e-3*tes_saturation_power/((1.71*0.1)**4.-0.1**4.)*4.*(1.71*0.1)**(4.-1.)

tesA = tessim.tes_dc_model(optical_loading_power = optical_loading_power, biasing_current = biasing_current, tes_saturation_power = tes_saturation_power, tes_leg_thermal_conductivity = tes_leg_thermal_conductivity, tes_heat_capacity = tes_heat_capacity)

t = np.linspace(0., time_sec, length)
P = np.ones_like(t) * tesA.optical_loading_power + np.sin(2 * np.pi * t * 1/20/60) * p_opt_pW / 1e12 * 1e-3
Ib = np.ones_like(t) * tesA.biasing_current
Tb = np.ones_like(t) * tesA.temperature_focal_plane
I = np.zeros_like(t)
T = np.zeros_like(t)

P = P + noise.add_white_noise(sigma, P)

I, T = tessim.TesDcRungeKuttaSolver(t, Ib, P, Tb, I, T, tesA)

Vb = tesA.biasing_current * (tesA.shunt_resistor * 0.5) / (tesA.shunt_resistor + 0.5)
SI = -1. / Vb

'''
plt.figure()
plt.title('Detector Input', fontsize=20)
plt.plot(t[10000:], P[10000:]-np.mean(P[10000:]), label = 'Input Power')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('time [s]', fontsize=20)
plt.ylabel('dP [W]', fontsize=20)
plt.legend(fontsize=12, loc=3)
plt.tight_layout()

plt.figure()
plt.title('Detector Output', fontsize=20)
plt.plot(t[10000:], I[10000:]-np.mean(I[10000:]), label = 'Output Current')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('time [s]', fontsize=20)
plt.ylabel('dI [A]', fontsize=20)
plt.legend(fontsize=12, loc=3)
plt.tight_layout()
'''

t = t[10000:]
dI = (I[10000:]-np.mean(I[10000:]))/SI
dP = P[10000:]-np.mean(P[10000:])

length = len(t)

dI_fft = sp.fft.fft(dI)
dP_fft = sp.fft.fft(dP)
freqs = sp.fft.fftfreq(length, 1/sampling_rate_hz)[:length//2]

dI_psd = dI_fft[0:length//2]*np.conjugate(dI_fft[0:length//2])/sampling_rate_hz/length
dP_psd = dP_fft[0:length//2]*np.conjugate(dP_fft[0:length//2])/sampling_rate_hz/length


plt.figure()
plt.title('NEP Test', fontsize=20)
plt.loglog(freqs, dP_psd**(1/2), label='input')
plt.loglog(freqs, dI_psd**(1/2), label='calibrated output')
plt.loglog(freqs, nep_aW_over_rthz / 1.e18*np.ones_like(freqs), 'k--', label = 'imo - 100 GHz LFT')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('frq [Hz]', fontsize=20)
plt.ylabel('PSD [W/sqrt(Hz)]', fontsize=20)
plt.legend(fontsize=12, loc=3)
plt.tight_layout()


plt.show()

