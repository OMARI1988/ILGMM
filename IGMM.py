import itertools
import numpy as np
from scipy import linalg
from GMM_BIC import *
from scipy.stats import ks_2samp
import numpy as np
from collections import namedtuple
import scipy


#----------------------------------------------------------------------------------------------------------------#
# a GMM based on BCC for a data X and a maximum number of components k and coverince types cv_types
def igmm(X,gmm_N,gmm_M):
	
	# initilizations od the W statitics test
	W_statistics = np.zeros(shape=(gmm_M.n_components,gmm_N.n_components))
	W_results = np.zeros(shape=(gmm_M.n_components,gmm_N.n_components))
	# initilizations od the Hotelling's T squared test
	H_statistics = np.zeros(shape=(gmm_M.n_components,gmm_N.n_components))
	H_results = np.zeros(shape=(gmm_M.n_components,gmm_N.n_components))
	# 4. assign each new data in X to a cluster in gmm_M
	Y_ = gmm_M.predict(X)
	# 5. for every component in gmm_M
	for k, (mean_M, covar_M) in enumerate(zip(gmm_M.means_, gmm_M.covars_)):
		# 6. Let Dk be the collection of all the data in component k.
		Dk = X[Y_==k]
		# 7. for every component in gmm_N
		for j, (mean_j, covar_j) in enumerate(zip(gmm_N.means_, gmm_N.covars_)):
			covar_j = fixing_covar(covar_j,gmm_N.covariance_type)
			# 8. calculate the W_statistic to determine if Dk has equal coverinece with covar_j
			W_statistics[k][j],W_results[k][j] = Covariance_Test(Dk,covar_j)
			# 9. if Dk passed the W statitics test.
			if W_results[k][j] == 1.0:
				# 10. Perform the Hotelling's T squared test to see if Dk has the same mean as mean_j
				H_statistics[k][j],H_results[k][j] = Mean_Test(Dk,mean_j)
				# 11. if Dk passed the Hotelling's T squared test
				if H_results[k][j] == 1.0:
					# 19 create a new component g N+M by merging j and k
					gmm_N.means_[j] = [1000,1000]
					print gmm_N.weights_
					print gmm_N.means_[j]
					print gmm_N.covars_[j]

	print gmm_N.n_components

	# Print the W statitics on the command window			
	print_W_statitics(W_results,gmm_M.n_components,gmm_N.n_components,'W')
	print_W_statitics(H_results,gmm_M.n_components,gmm_N.n_components,'H')

	return gmm_N

#----------------------------------------------------------------------------------------------------------------#
# 3.1 Testing for equality to a covariance matrix
def Covariance_Test(x,covar):
	# finding Lo (lower triangular matrix obtained by Cholesky decomposition of covar)
	Lo = np.linalg.cholesky(covar)
	Lo_inv = np.linalg.inv(Lo)
	d = len(Lo_inv)
	# finding yi
	yi = {}
	for i in x:
		y = np.dot(Lo_inv,np.vstack(i))
		for n in range(d):
			if n in yi:
				yi[n].append(y[n][0])
			else:
				yi[n] = [y[n][0]]
	# Computing the sample covariance matrix of yi, Sy
	Sy = np.zeros((d,d),dtype=np.float)
	for i in yi:
		Sy[i,i] = np.cov(yi[i])
	# computing the trace of a matrix Sy-I
	I = np.identity(d)
	c1 = np.matrix.trace((Sy - I)**2)/float(d)
	# compute the trace of matix Sy
	n = float(len(x))	# n is the number of the points in kluster k
	c2 = ( float(d)/n ) * ( 1/float(d) * np.matrix.trace(Sy) )**2
	# compute W
	W = c1 - c2 + float(d)/n
	# perform the test
	alpha = .05
	test = n*W*d/2.0
	p_value = scipy.stats.chi2.pdf(test, (d*(d+1))/2)
	result = 0
	if p_value>alpha:
		result = 1.0
	return test,result

#----------------------------------------------------------------------------------------------------------------#
# 3.2 Testing for equality to a mean vector
def Mean_Test(x,mean):
	# compute sample mean
	d = len(x[0])
	n = len(x)
	x_mean = []
	for i in range(d):
		x_mean.append(np.mean(x[:,i]))
	# compute the sample covariance
	S = np.cov(x.T)
	S_inv = np.linalg.inv(S)
	# computing the T squared test
	c1 = np.transpose([mean - x_mean])
	T = n*np.dot(np.dot(np.transpose(c1),S_inv),c1)
	F = T[0][0]*float(n-d)/float(d*(n-1))
	alpha = 0.05 #Or whatever you want your alpha to be.
	p_value = scipy.stats.f.pdf(F, d, n-d)
	result = 0
	if p_value>alpha:
		result = 1.0
	return F,result

#----------------------------------------------------------------------------------------------------------------#
# fixing the covariance matrix
def fixing_covar(covar_j,ctype):
	if ctype != 'full':
		N = len(covar_j)
		covar = np.zeros((N,N),dtype=np.float)
		for i, c in enumerate(covar_j):
			covar[i,i] = c
		covar_j = covar
	return covar_j

#----------------------------------------------------------------------------------------------------------------#
# Print the W_statistics matrix on command window
def print_W_statitics(W,M,N,msg):
	print
	print('***************************************************')
	if msg == 'W':
		print('W statistics for %d new clusters and %d old clusters.' %(M,N))
	if msg == 'H':
		print('Hotelling test for %d new clusters and %d old clusters.' %(M,N))
	print('***************************************************')
	print
	rows = ['W_statistics']
	for i in range(N):
		rows.append('old_'+str(i))
	Row = namedtuple('Row',rows)

	all_data = []
	for i in range(M):
		d = {'W_statistics': 'new_'+str(i)}
		for j in range(N):
			d['old_'+str(j)] = str(W[i][j])
		data = Row(**d)
		all_data.append(data)
	pprinttable(all_data)
	print

#----------------------------------------------------------------------------------------------------------------#
# The function that does the printing
def pprinttable(rows):
	if len(rows) > 1:
		headers = rows[0]._fields
		lens = []
		for i in range(len(rows[0])):
			lens.append(len(max([x[i] for x in rows] + [headers[i]],key=lambda x:len(str(x)))))
		formats = []
		hformats = []
		for i in range(len(rows[0])):
			if isinstance(rows[0][i], int):
				formats.append("%%%dd" % lens[i])
			else:
				formats.append("%%-%ds" % lens[i])
			hformats.append("%%-%ds" % lens[i])
		pattern = " | ".join(formats)
		hpattern = " | ".join(hformats)
		separator = "-+-".join(['-' * n for n in lens])
		print hpattern % tuple(headers)
		print separator
		for line in rows:
			print pattern % tuple(line)
	elif len(rows) == 1:
		row = rows[0]
		hwidth = len(max(row._fields,key=lambda x: len(x)))
		for i in range(len(row)):
			print "%*s = %s" % (hwidth,row._fields[i],row[i])
























	




