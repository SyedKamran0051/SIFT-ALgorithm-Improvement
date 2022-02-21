import numpy as np
from scipy.ndimage.filters import convolve

from gaussian_filter import gaussian_filter 

def generate_octave(init_level, s, sigma):  #to generate a Gaussian octave, we simply need to select sigma
	octave = [init_level]

	k = 2**(1/s)
	kernel = gaussian_filter(k * sigma)

	for i in range(s+2):                    #repeatedly convolve with this Gaussian filter
		next_level = convolve(octave[-1], kernel)
		octave.append(next_level)

	return octave

def generate_gaussian_pyramid(im, num_octave, s, sigma):   #generate the whole Gaussian pyramid
	   #s of 5 is used 
    pyr = []

    for _ in range(num_octave):
        octave = generate_octave(im, s, sigma)
        pyr.append(octave)
        im = octave[-3][::2, ::2]

    return pyr