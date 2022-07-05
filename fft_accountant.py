import numpy as np

def fft_ConvN(l_grid,fL_array,composition):
    l_grid_padding = np.arange(composition*min(l_grid),composition*max(l_grid),l_grid[1]-l_grid[0])
    
    # We can actually avoid all FFT shifting (and normalization) if we zero-pad at the end of the vector, 
    # rather than on both sides. In fact the fft function will automatically do this 
    # if we just add a second argument with the desired length.
    fFFT = np.fft.fft(fL_array, len(l_grid_padding))
    cfFFT_array = np.power(fFFT,composition)
    fLConvFFT_array = np.ifft(cfFFT_array)
    fLConvFFT_array = np.max(0,fLConvFFT_array)
    return l_grid_padding, fLConvFFT_array