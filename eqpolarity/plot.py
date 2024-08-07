def plot_confusionmatrix(cf=None,categories=['Up','Down'],figname='Conf_Matrix_before_transferlearning.png',ifshow=True):
	'''
	plot_confusionmatrix: plot confusion matrix for an arbitrary classification problem
	
	INPUT
	cf: confusion matrix (2x2 or 3x3)
	
	EXAMPLE
	demos/test_texas_transferlearning.py
	
	'''
	import numpy as np
	import seaborn as sn
	import matplotlib.pyplot as plt
	font = {'family' : 'normal',
		'weight' : 'bold',
		'size'   : 16}
	plt.rc('font', **font)

	# cf = cf_matrix

	# import numpy as np
	# cf=np.array([[13596,   947],
	#		[  296,  8141]])

	# after:
	# cf=np.array([[14373,   170],
	#  [  158,  8279]])
 
	group_percentages = []
	counts = []
	for i in range(len(cf)):
		for j in range(len(cf)):
			group_percentages.append(cf[j, i]/np.sum(cf[:, i]))
			counts.append(cf[j, i])

	percentages_matrix = np.reshape(group_percentages, (2, 2))
	group_percentages = ['{0:.2%}'.format(value) for value in group_percentages]

	labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_percentages, counts)]
	labels = np.asarray(labels).reshape(2, 2, order = 'F')

	fig = plt.figure(figsize=(10,7))
	sn.set(font_scale=2) # for label size
	sn.heatmap(percentages_matrix, annot = labels, fmt = '', xticklabels=categories, yticklabels = categories, cbar = False)
	fig.savefig(figname,bbox_inches='tight',transparent=True, dpi =100)
	if ifshow:
		plt.show()
	else:
		plt.close()
