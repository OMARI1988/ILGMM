import itertools
import numpy as np
from scipy import linalg
from GMM_BIC import *


# a GMM based on BCC for a data X and a maximum number of components k and coverince types cv_types
def igmm(X,gmm_N,gmm_M):
	
	# 4. assign each new data in X to a cluster in gmm_M
	Y_ = gmm_M.predict(X)
	print Y_

	# 5. for every component in gmm_M
	for k, (mean_M, covar_M) in enumerate(zip(gmm_M.means_, gmm_M.covars_)):

		# 7. for every component in gmm_N
		for j, (mean_N, covar_N) in enumerate(zip(gmm_N.means_, gmm_N.covars_)):

			# 8. test if covar_M and covar_N are the same
			print covar_M
			print covar_N
			print '---------------'


	return gmm_N


