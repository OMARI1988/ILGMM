
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
sample_size = 20			# how many frames in each sample
initial_size = 20			# how many frames in each sample
size = 5000
k = 8
directory = '/home/omari/Python_images/ILGMM/test2/'
cv_types = ['spherical', 'tied', 'diag', 'full']
#cv_types = ['full', 'diag']
np.random.seed(11234)
plt.ion()

#######################################
#----------- Apply GMM-BIC -----------#
#######################################
mean1 = [0,0]
cov1 = [[3,.5],[-.7,10]] 

mean2 = [30,-10]
cov2 = [[12,4],[1,7]] 

mean3 = [30,30]
cov3 = [[3,1],[0,10]]

mean4 = [10,10]
cov4 = [[1,0],[.3,10]]

GT_mean = [mean1, mean2, mean3, mean4]
GT_cov = [cov1, cov2, cov3, cov4]
GT = [GT_mean, GT_cov]

X1 = np.random.multivariate_normal(mean1,cov1,initial_size)
X2 = np.random.multivariate_normal(mean2,cov2,initial_size)
X3 = np.random.multivariate_normal(mean3,cov3,initial_size)
X = np.vstack([X1,X2,X3])
#X = np.vstack([X1,X3])


gmm_N, bic_N = gmm_bic(X, k, ['full'])
N = float(len(X))
###################################
#----------- main loop -----------#
###################################
for frame_number in range(size/sample_size):
	
	###############################################
	#----------- build the data vector -----------#
	###############################################
	X = []
	X1 = np.random.multivariate_normal(mean1,cov1,sample_size)
	X2 = np.random.multivariate_normal(mean2,cov2,sample_size)
	X3 = np.random.multivariate_normal(mean3,cov3,sample_size)
	X4 = np.random.multivariate_normal(mean4,cov4,sample_size)
	#X = np.vstack([X1,X2,X3])
	X = np.vstack([X1,X2,X3,X4])
	#X = np.vstack([X1,X3])
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
	plot_data(X, gmm_N, bic, k, cv_types, GT, fig_no)
	#plot_data(X, gmm_M, bic, k, cv_types, 2)
	plt.draw()
	#plt.savefig(directory+str(frame_number*sample_size)+'.png')
	print
	print '---------------------------'
	print '-- frame number : ',frame_number*sample_size ,'   --'
	print '---------------------------'
	print 











