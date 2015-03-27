
import numpy as np
from GMM_BIC import *
from GMM_functions import *

###############################
#----------- Notes -----------#
###############################
# sample size 'sample_size' should have a minimum value of 2

#######################################
#----------- initilization -----------#
#######################################
hyp = read_file('16_colors')		# read a pickle file
word = 'red'				# test for this word
sample_size = 2000				# how many frames in each sample
k = 16
cv_types = ['spherical', 'tied', 'diag', 'full']



Results = {}

###################################
#----------- main loop -----------#
###################################
for word in hyp['hyp']:

	###############################################
	#----------- build the data vector -----------#
	###############################################
	frames = []
	for frame in hyp['hyp'][word]['point_HSV_x']:
		frames.append(frame)

	for frame_number in range(0,len(frames),sample_size):
	
		X = []
		for j in range(sample_size):
			if frame_number+j > len(frames)-1:
				break
			else:	
				frame = frames[frame_number+j]
			x_val = hyp['hyp'][word]['point_HSV_x'][frame]
			y_val = hyp['hyp'][word]['point_HSV_y'][frame]
			z_val = hyp['hyp'][word]['point_HSV_z'][frame]
			for num, (x, y, z) in enumerate(zip(x_val, y_val, z_val)):
				if X == []:
					X = [[x/200.0,y/200.0,z/100.0]]
				else:
					X = np.vstack([X,[x/200.0,y/200.0,z/100.0]])

		#######################################
		#----------- Apply GMM-BIC -----------#
		#######################################
		best_gmm, bic = gmm_bic(X, k, cv_types)

		######################################
		#----------- Test GMM-BIC -----------#
		######################################
		result = test_gmm(X, best_gmm, hyp, word)

		######################################
		#----------- PLOT results -----------#
		######################################
		plot_data_winner(X, best_gmm, bic, k, cv_types, 1, result, word)
		Results[word] = result['score']
		plt.show()
		print Results










