# Privacy Accountants with DP-SGD

## mnist_dpgsd_test_keras_vectorized.py
This is the main function of the experiments, with keras vectorized dp-sgd optimizers modfied by different adding-noise mechanisms.
Comparing with the one in the tutorial, we added two new flags, the "dpsgd_type" and "logging", which allow users to switch between dpsgd types (i.e., gaussian, laplace, cactus).

## dp_optimizer_keras_vectorized.py
You may find a file with the same name under tensorflow_privacy/privacy/optimizers. It worth noting that the noise adding mechanism is based on a tf.function decorator to allow we call numpy function in the graph mode. The cactus generating function is based on a pool with large number of samples associated with different parameters. The cactus sample generating function is located a jupyter notebook which will be introduced later.

## cactus_generating.py
This file and functions in this file is somehow "independent" from all others. As the pdf generator works significantly slower than its sibling on Matlab. You may directly run it and it will generate "x" and the corresponding cdf files, they are saved in .cvs files under test_data directory.

## cactus_sampling.py
This sampling function relies on the .csv files which give the 'cdf' on range 'x'(from -8 to 8, quantization level=1/200) with a given variance, and a pool of samples will be saved to a .npy file.
NOTE: The file name contains the variance but NOT the stddev! It worth noting that in the main codes, we use stddev which is defined as "noise_multiplier*l2_norm_clip".

## cactus_sampling_generating.ipynb
This notebook combines "cactus_generating.py" and "cactus_sampling.py".

## privacy_plot.ipynb
This notebook contains ploting functions to display the collected results.

## privacy_test.ipynb
This notebook is a testing case for functions used in other codes.
