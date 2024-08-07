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



