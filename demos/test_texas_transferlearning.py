## First download Texas testing data in 
# https://mega.nz/file/chxx1Z5Y#zXNRKT5aeNy7AGREKEUIq71TREK8hcUyXA1ZOkQ9DlM


## Import data
import numpy as np
datall = np.load('./TexasData/datall_Texas.npy')
polall = np.load('./TexasData/polall_Texas.npy')

## Load EQpolarity model
from eqpolarity.utils import construct_model
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
from eqpolarity import plot_confusionmatrix
# import numpy as np
# cf_matrix=np.array([[13596,   947],
#        [  296,  8141]])
#after:
# cf_matrix=np.array([[14373,   170],
#  [  158,  8279]])
plot_confusionmatrix(cf=cf_matrix,categories=['Up','Down'],figname='Conf_Matrix_before_transferlearning.png',ifshow=False)

# weightname='best_weigths_Binary_Texas_Transfer10.weights.h5'
import datetime
today=datetime.date.today()
weightname='best_weigths_Binary_Texas_Transfer10_%s.weights.h5'%str(today)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


checkpoint = ModelCheckpoint(filepath='best_weigths_Binary_Texas_Transfer10.weights.h5',
                             monitor='val_acc',
                             mode = 'max',
                             verbose=1,
                             save_weights_only=True,
                             save_best_only=True)

lr_reducer = ReduceLROnPlateau(factor=0.1,
                                   cooldown=0,
                                   patience=50,
                                   min_lr=0.5e-6,
                                   monitor='val_acc',
                                   mode = 'max',
                                  verbose= 1)
                                  
ind = np.random.permutation(len(datall))
a = int(10*len(ind)/100)
ind = ind[0:a]
x = datall[ind]
y = polall[ind]

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=['binary_crossentropy'], metrics=['acc'])
model.load_weights('best_weigths_Binary_CSCN_Best.h5')

model.fit(x, y, batch_size=128, epochs=50, verbose =1, validation_split=0.1, shuffle=True, callbacks=[checkpoint,lr_reducer])


# try new model
model.load_weights(weightname)
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

## print scores
accuracy_score(labtest,outtest),precision_score(labtest,outtest, average='micro'),recall_score(labtest,outtest, average='micro'),f1_score(labtest,outtest, average='micro')
accuracy_score(labtest,outtest),precision_score(labtest,outtest, average=None),recall_score(labtest,outtest, average=None),f1_score(labtest,outtest, average=None)
accuracy_score(labtest,outtest),precision_score(labtest,outtest, average='macro'),recall_score(labtest,outtest, average='macro'),f1_score(labtest,outtest, average='macro')

cf_matrix = confusion_matrix(labtest, outtest)
print(cf_matrix)

#after:
# cf_matrix=np.array([[14373,   170],
#  [  158,  8279]])
plot_confusionmatrix(cf=cf_matrix,categories=['Up','Down'],figname='Conf_Matrix_after_transferlearning.png',ifshow=False)

