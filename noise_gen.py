import numpy as np
import math
import scipy as sp

# Add white noise component
def add_white_noise(sigma, data):
	'''Add a white noise component to a data vector:
	sigma = noise standard deviation
	data = noisless data
	'''
	noise = np.random.normal(0., sigma, data.shape)
	
	return noise
	
# Add 1/f noise component below fknee and white noise above fknee
def add_pink_noise(fknee_hz, alpha, sigma, freq_hz, data):
	''' Add a 1/f component to a data vector:
	fknee_hz = 1/f knee frequency in hz
	alpha = 1/f steepness
	sigma = noise standard deviation (white noise)
	freq_hz = sampling rate in hz
	data = noisless data
	'''
	
	# find first power of 2 > len(data)
	length = pow(2, math.ceil(math.log(len(data), 2)))

	# white noise of sigma = 1
	noise = np.random.normal(0, 1, length)

	# fft calculation of white noise
	fourier_trans = sp.fft.fft(noise, n = length)
	freqs = sp.fft.fftfreq(length, d = 1/freq_hz)

	# filter the white noise in the frequency domain with 1/f filter
	filtered = np.zeros_like(freqs)
	for i in range(len(freqs)):
		
		if freqs[i] != 0:
			
			filtered[i] = np.sqrt((1 + pow(abs(freqs[i]) / (fknee_hz), -1 * alpha))) * sigma
		
		elif freqs[i]:
		
			filtered[i] = 0

	# back to time domain
	inverse_fft = sp.fft.ifft(fourier_trans * filtered)
	
	# return real part of original len(data)
	noise = np.real(inverse_fft[:len(data)])
	
	return noise
