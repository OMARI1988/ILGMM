
import numpy as np
from GMM_BIC import *
from IGMM import *
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
sample_size = 20			# how many frames in each sample
initial_size = 20			# how many frames in each sample
k = 16
#cv_types = ['spherical', 'tied', 'diag', 'full']
cv_types = ['full']

frames = []
for frame in hyp['hyp'][word]['point_HSV_x']:
	frames.append(frame)


#######################################
#----------- Apply GMM-BIC -----------#
#######################################
for frame_number in range(initial_size):
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

gmm_N, bic_N = gmm_bic(X, k, cv_types)
N = float(len(X))

###################################
#----------- main loop -----------#
###################################
for frame_number in range(initial_size,len(frames),sample_size):
	
	###############################################
	#----------- build the data vector -----------#
	###############################################
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
	M = float(len(X))

	########################################
	#----------- Apply iGMM-BIC -----------#
	########################################
	gmm_M, bic = gmm_bic(X, k, cv_types)
	gmm_NM = igmm(X, gmm_N, gmm_M, N, M)


	##########################################
	#----------- Update variables -----------#
	##########################################
	N += M
	gmm_N = gmm_NM


	######################################
	#----------- PLOT results -----------#
	######################################
	fig_no = 1
	plot_data(X, gmm_N, bic, k, cv_types, 0, fig_no)
	#plot_data(X, gmm_N, bic_N, k, cv_types, 1)
	#plot_data(X, gmm_M, bic, k, cv_types, 2)
	plt.show()










