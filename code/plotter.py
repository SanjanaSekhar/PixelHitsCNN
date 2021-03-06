import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def plot_cnn_loss(history,label,img_ext):
	plt.plot(history['%s_loss'%(label)])
	plt.plot(history['val_%s_loss'%(label)])
	plt.title('%s position - model loss'%(label))
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['%s-train'%(label), '%s-validation'%(label)], loc='upper right')
	#plt.show()
	plt.savefig("plots/loss_%s_%s.png"%(label,img_ext))
	plt.close()

def plot_dnn_loss(history,label,img_ext):
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('%s position - model loss'%(label))
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['%s-train'%(label), '%s-validation'%(label)], loc='upper right')
	#plt.show()
	plt.savefig("plots/loss_%s_%s.png"%(label,img_ext))
	plt.close()


def plot_residuals(residuals,mean,sigma,RMS,label,img_ext):
	plt.hist(residuals, bins=np.arange(-300,300,0.25), histtype='step', density=True,linewidth=2, label=r'$\vartriangle$'+label)
	xmin, xmax = plt.xlim()
	x = np.linspace(xmin, xmax, 100)
	p = norm.pdf(x, mean, sigma)
	plt.title('residuals in %s, RMS = %0.2f, %s = %0.2f'%(label,RMS,r'$\sigma$',sigma))
	#plt.ylabel('No. of samples')
	plt.xlabel(r'$\mu m$')

	plt.plot(x, p, 'k', linewidth=1,color='red',label='gaussian fit')
	plt.legend()
	plt.savefig("plots/residuals_%s_%s.png"%(label,img_ext))
	plt.close()

def plot_by_clustersize(residuals,clustersize,label,img_ext):

	residuals = np.asarray(residuals)
	clustersize = np.asarray(clustersize)
	max_size = int(np.amax(clustersize))
	sigma_per_size = np.zeros((max_size+1,1))
	print(residuals.shape)
	pp = PdfPages('plots/res_%s_csize_%s.pdf'%(label,img_ext))

	for i in range(1,max_size+1):
		indices = np.argwhere(clustersize==i)[:,0]
		print(i,':',indices.shape)
		if(len(indices)==0):
			sigma_per_size[i]=0.
		else:
			residuals_per_size = residuals[indices]
			sigma_per_size[i] = np.std(residuals_per_size)
			plt.hist(residuals_per_size, bins=np.arange(-300,300,0.25), histtype='step',linewidth=2, label=r'$\vartriangle$'+label)
			plt.xlabel(r'$\mu m$')
			plt.title('residuals in %s for clustersize = %i, %s = %0.2f'%(label,i,r'$\sigma$',sigma_per_size[i]))
			pp.savefig()
			plt.close()

	pp.close()

	x = np.linspace(1, max_size, max_size)
	print(np.amin(clustersize),max_size)
	print(x.shape,sigma_per_size[1:].shape)
	plt.scatter(x,sigma_per_size[1:])
	plt.xlabel('clustersize in %s'%(label))
	plt.ylabel('resolution in %s'%(label))
	plt.title('resolution vs clustersize in %s'%(label))
	plt.savefig("plots/resvssize_%s_%s"%(label,img_ext))
	plt.close()

