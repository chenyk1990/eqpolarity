## First download Texas testing data in 
# https://mega.nz/file/chxx1Z5Y#zXNRKT5aeNy7AGREKEUIq71TREK8hcUyXA1ZOkQ9DlM


## Import data
import numpy as np
datall = np.load('./TexasData/datall_Texas.npy')
polall = np.load('./TexasData/polall_Texas.npy')

## Load EQpolarity model
from eqpolarity import construct_model
input_shape = (600,1)
model=construct_model(input_shape)
model.summary()

## Load pre-trained model for prediction
model.load_weights('best_weigths_Binary_CSCN_Best.h5')
out = model.predict(datall,batch_size=1024, verbose=1)

## Applying threshold
#outtest = np.argmax(out,axis=-1)
thre = 0.5
outtest = out
outtest[outtest<thre]=0
outtest[outtest>=thre]=1
labtest = polall


## Calculating accuracy, precision, recall, and F1-score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_score(labtest,outtest),precision_score(labtest,outtest,average='micro'),recall_score(labtest,outtest,average='micro'),f1_score(labtest,outtest, average='micro')

accuracy_score(labtest,outtest),precision_score(labtest,outtest, average=None),recall_score(labtest,outtest, average=None),f1_score(labtest,outtest, average=None)

#Generate the confusion matrix
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(labtest, outtest)
print(cf_matrix)



## plotting confusion matrix
import seaborn as sn
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
plt.rc('font', **font)

cf = cf_matrix
categories=['Up','Down']
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
fig.savefig('Conf_Matrix',bbox_inches='tight',transparent=True, dpi =100)
plt.show()


# 
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# 
# 
# checkpoint = ModelCheckpoint(filepath='best_weigths_Binary_Texas_Transfer10.weights.h5',
#                              monitor='val_acc',
#                              mode = 'max',
#                              verbose=1,
#                              save_weights_only=True,
#                              save_best_only=True)
# 
# lr_reducer = ReduceLROnPlateau(factor=0.1,
#                                    cooldown=0,
#                                    patience=50,
#                                    min_lr=0.5e-6,
#                                    monitor='val_acc',
#                                    mode = 'max',
#                                   verbose= 1)
#                                   
# ind = np.random.permutation(len(datall))
# a = int(10*len(ind)/100)
# ind = ind[0:a]
# x = datall[ind]
# y = polall[ind]
# 
# model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=['binary_crossentropy'], metrics=['acc'])
# model.load_weights('best_weigths_Binary_CSCN_Best.h5')
# 
# model.fit(x, y, batch_size=128, epochs=50, verbose =1, validation_split=0.1, shuffle=True, callbacks=[checkpoint,lr_reducer])


# try new model
model.load_weights('best_weigths_Binary_Texas_Transfer10.weights.h5')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
out = model.predict(datall,batch_size=1024, verbose=1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#outtest = np.argmax(out,axis=-1)
thre = 0.5
outtest = out
outtest[outtest<thre]=0
outtest[outtest>=thre]=1
labtest = polall
accuracy_score(labtest,outtest),precision_score(labtest,outtest, average='micro'),recall_score(labtest,outtest, average='micro'),f1_score(labtest,outtest, average='micro')


