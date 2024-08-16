## Import data
#polall is polarity label from Texas (Size: 22980x1)
#datall is 600-sample waveform data (Z-component) from Texas (Size: 22980 x 600 x 1); the waveform is centered by P-arrival (manual from Texas analysts) sample

import numpy as np

# datall_Texas.npy can also be downloaded from https://mega.nz/file/chxx1Z5Y#zXNRKT5aeNy7AGREKEUIq71TREK8hcUyXA1ZOkQ9DlM
# datall = np.load('../data/TexasData/datall_Texas.npy') 
polall = np.load('../data/TexasData/polall_Texas.npy')

data=[]
for ii in range(6):
    data.append(np.load('../data/TexasData/datall_Texas%d.npy'%(ii+1)))
datall=np.concatenate(data,axis=0)

## plot waveforms and polarities
nwaveforms=5;

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 10))

for ii in range(nwaveforms):
	ax=plt.subplot(nwaveforms,1,ii+1)
	plt.plot(np.linspace(0,600,600),datall[ii,:,:],linewidth=2,color='k')
	plt.title('Label='+str(polall[ii])+' Polarity: '+{'0':'UP','1':'Down'}[str(polall[ii])])
	plt.ylabel('Amplitude')
	ax.set_yticks([])
	if ii<nwaveforms-1:
		ax.set_xticks([])
plt.xlabel('Sample #');

plt.savefig('test_plot_waveforms_and_polarity.png',format='png',dpi=300)
plt.show()
	
	


