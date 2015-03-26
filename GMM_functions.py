from mpl_toolkits.mplot3d import Axes3D
import colorsys
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import numpy as np
import pickle

def read_file(x):

	directory = '/home/omari/ros_ws/src/baxter_demos/share/'
	pkl_file = open(directory+x+'.pkl', 'rb')
	print ' - loading data..'
	data1 = pickle.load(pkl_file)
	print ' - file loaded..'
	hyp = {}
	hyp['valid_HSV_hyp'] = []
	hyp['valid_dis_hyp'] = []
	hyp['valid_dir_hyp'] = []
	POINTS_HSV = data1['HSV']
	POINTS_SPA = data1['SPA']
	hyp = data1['hyp']
	return hyp

def find_RGB_map(N):
	HSV_tuples = [(x*1.0/N, 1, 1) for x in range(N)]
	return map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

#######################################
#----------- plot function -----------#
#######################################
# plot function takes as input the Data vector, the best GMM, the BIC results, the maximum components number, and the covariance types
def plot_data(X, best_gmm, bic, k, cv_types, fig):
	plt.figure(fig)

	n_samples = len(X)
	if k>(n_samples+1):
		k = n_samples+1

	n_components_range = range(1, k)

	bic = np.array(bic)
	color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
	clf = best_gmm
	bars = []

	# Plot the BIC scores
	spl = plt.subplot(2, 1, 1)
	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
		xpos = np.array(n_components_range) + .2 * (i - 2)
		bars.append(plt.bar(xpos, bic[i * len(n_components_range): (i + 1) * len(n_components_range)], width=.2, color=color))
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bic.argmin() / len(n_components_range))
	plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
	spl.set_xlabel('Number of components')
	spl.legend([b[0] for b in bars], cv_types)

	Best_number_of_clusters = np.mod(bic.argmin(), len(n_components_range)) + 1

	print Best_number_of_clusters

	color_iter = find_RGB_map(Best_number_of_clusters)
	# Plot the winner
	splot = plt.subplot(2, 1, 2, projection='3d')
	Y_ = clf.predict(X)
	for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covars_, color_iter)):
		if not np.any(Y_ == i):
			continue
		splot.scatter([r for r in X[Y_ == i, 0]], [r for r in X[Y_ == i, 1]], [r for r in X[Y_ == i, 2]], c=color, marker='o')


	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.xticks(())
	plt.yticks(())
	plt.title('Selected GMM: full model, '+str(Best_number_of_clusters)+' components')
	plt.subplots_adjust(hspace=.35, bottom=.02)











