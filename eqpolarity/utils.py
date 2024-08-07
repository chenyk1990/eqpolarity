import obspy
import numpy as np
import h5py
import glob
import math
import os
import shutil
from scipy import signal
from scipy.signal import butter, lfilter
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dropout, Reshape 
from tensorflow.keras.layers import Bidirectional, concatenate, BatchNormalization, ZeroPadding1D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.callbacks import CSVLogger
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from tensorflow.python.keras.layers import Layer, InputSpec
#import tensorflow as tf
#tf.compat.v1.disable_v2_behavior
#tf.compat.v1.reset_default_graph()
#tf.compat.v1.enable_eager_execution
#tf.enable_eager_execution()

from sklearn.metrics import accuracy_score

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Add, Bidirectional, Conv1D, Dense, Dropout, Embedding, Flatten, Reshape, multiply
from keras.layers import concatenate, GRU, Input, LSTM, MaxPooling1D
from keras.layers import GlobalAveragePooling1D,  GlobalMaxPooling1D, SpatialDropout1D
from keras.models import Model
# from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.models import Model

def mlp(x, hidden_units, dropout_rate):
	for units in hidden_units:
		x = layers.Dense(units, activation=tf.nn.gelu)(x)
		#x = layers.Dense(units, activation='relu')(x)
		x = layers.Dropout(dropout_rate)(x)
	return x
	
drop_rate = 0.2
stochastic_depth_rate = 0.1

w1 = 100

positional_emb = False
conv_layers = 2
num_classes = 1
input_shape = (600,1)
image_size = 600  # We'll resize input images to this size
projection_dim = int(2*w1)
num_heads = 4
transformer_units = [
	projection_dim,
	projection_dim,
]  # Size of the transformer layers
transformer_layers = 4

#Below is for 1D CNN
def CBP(yx,F,K):
    x = Conv1D(F,K,padding='same', activation='relu')(yx)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    return x
    
#Below is for VGG
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D

def VGG_16(weights_path=None,img_shape=(600,1)):
    model = Sequential()
    model.add(ZeroPadding1D((1),input_shape=img_shape))
    model.add(Conv1D(8, 3, 1, activation='relu'))
    model.add(ZeroPadding1D((1)))
    model.add(Conv1D(8, 3, 1, activation='relu'))
    model.add(MaxPooling1D((2), strides=(2)))

    model.add(ZeroPadding1D((1)))
    model.add(Conv1D(16, 3, 1, activation='relu'))
    model.add(ZeroPadding1D((1)))
    model.add(Conv1D(16, 3, 1, activation='relu'))
    model.add(MaxPooling1D((2), strides=(2)))

    model.add(ZeroPadding1D((1)))
    model.add(Conv1D(32, 3, 1, activation='relu'))
    model.add(ZeroPadding1D((1)))
    model.add(Conv1D(32, 3, 1, activation='relu'))
    model.add(ZeroPadding1D((1)))
    model.add(Conv1D(32, 3, 1, activation='relu'))
    model.add(MaxPooling1D((2), strides=(2)))



    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    if weights_path:
        model.load_weights(weights_path)

    return model


#Below is for Alexnet
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D
def alexnet_model(img_shape=(600, 1), n_classes=10, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv1D(16, (111), input_shape=img_shape,
		padding='same', kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling1D(pool_size=(2)))

	# Layer 2
	alexnet.add(Conv1D(32, (5), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling1D(pool_size=(2)))

	# Layer 3
	alexnet.add(ZeroPadding1D((1)))
	alexnet.add(Conv1D(64, (3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling1D(pool_size=(2)))

	# Layer 4
	alexnet.add(ZeroPadding1D((1)))
	alexnet.add(Conv1D(128, (3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))

	# Layer 5
	alexnet.add(ZeroPadding1D((1)))
	alexnet.add(Conv1D(128, (3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling1D(pool_size=(2)))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(256))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(256))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(Dense(1))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('sigmoid'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet
	
def jday_to_mmdd(jday, year):
	"""
	Convert day of year (jday) to MM/DD format
	"""
	days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
	if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
		days_in_month[1] = 29  # Leap year
	else:
		days_in_month[1] = 28  # Non-leap year
	
	days_so_far = 0
	month = 0
	for days in days_in_month:
		days_so_far += days
		month += 1
		if days_so_far >= jday:
			day = jday - (days_so_far - days)
			return "{month:02d}/{day:02d}".format(month=month, day=day)

import tensorflow as tf
class CCTTokenizer1(layers.Layer):
	def __init__(
		self,
		kernel_size=4,
		stride=1,
		padding=1,
		pooling_kernel_size=3,
		pooling_stride=(2,2,2,2,2,2,2,2),
		num_conv_layers=conv_layers,
		num_output_channels=[int(projection_dim), int(projection_dim), int(projection_dim), int(projection_dim), int(projection_dim), int(projection_dim), int(projection_dim), int(projection_dim)],
		positional_emb=positional_emb,
		**kwargs,
	):
		super(CCTTokenizer1, self).__init__(**kwargs)

		# This is our tokenizer.
		self.conv_model = tf.keras.Sequential()
		for i in range(num_conv_layers):
			self.conv_model.add(
				layers.Conv1D(
					num_output_channels[i],
					kernel_size,
					stride,
					padding="same",
					use_bias=False,
					activation="relu",
					kernel_initializer="he_normal",
				)
			)
			#self.conv_model.add(layers.ZeroPadding2D(padding))
			self.conv_model.add(
				layers.MaxPool1D(pooling_kernel_size, (pooling_stride[i]), "same")
			)

		self.positional_emb = positional_emb

	def call(self, images):
		outputs = self.conv_model(images)
		# After passing the images through our mini-network the spatial dimensions
		# are flattened to form sequences.
		reshaped = tf.reshape(
			outputs,
			(-1, tf.shape(outputs)[1], tf.shape(outputs)[-1]),
		)
		return reshaped

	def positional_embedding(self, image_size):
		# Positional embeddings are optional in CCT. Here, we calculate
		# the number of sequences and initialize an `Embedding` layer to
		# compute the positional embeddings later.
		if self.positional_emb:
			dummy_inputs = tf.ones((1, image_size, 1))
			dummy_outputs = self.call(dummy_inputs)
			sequence_length = dummy_outputs.shape[1]
			projection_dim = dummy_outputs.shape[-1]

			print(dummy_outputs,sequence_length,projection_dim)
			embed_layer = layers.Embedding(
				input_dim=sequence_length, output_dim=projection_dim
			)
			return embed_layer, sequence_length
		else:
			return None
			
# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
	def __init__(self, drop_prop, **kwargs):
		super(StochasticDepth, self).__init__(**kwargs)
		self.drop_prob = drop_prop

	def call(self, x, training=None):
		if training:
			keep_prob = 1 - self.drop_prob
			shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
			random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
			random_tensor = tf.floor(random_tensor)
			return (x / keep_prob) * random_tensor
		return x
		  
def create_cct_model1(inputs):


	# Augment data.
	#augmented = data_augmentation(inputs)

	# Encode patches.
	cct_tokenizer = CCTTokenizer1()
	encoded_patches = cct_tokenizer(inputs)

	# Apply positional embedding.
	if positional_emb:
		pos_embed, seq_length = cct_tokenizer.positional_embedding(image_size)
		positions = tf.range(start=0, limit=seq_length, delta=1)
		position_embeddings = pos_embed(positions)
		encoded_patches += position_embeddings

	# Calculate Stochastic Depth probabilities.
	dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

	# Create multiple layers of the Transformer block.
	for i in range(transformer_layers):
		# Layer normalization 1.
		x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

		# Create a multi-head attention layer.
		attention_output = layers.MultiHeadAttention(
			num_heads=num_heads, key_dim=projection_dim, dropout=0.2
		)(x1, x1)

		#print(encoded_patches)
		# Skip connection 1.
		attention_output = StochasticDepth(dpr[i])(attention_output)
		x2 = layers.Add()([attention_output, encoded_patches])

		# Layer normalization 2.
		x3 = layers.LayerNormalization(epsilon=1e-5)(x2)
		#x3 = x2
		
		# MLP.
		x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.2)

		# Skip connection 2.
		#print(x3)
		x3 = StochasticDepth(dpr[i])(x3)
		#print(x3)
		encoded_patches = layers.Add()([x3, x2])
	 
	# Apply sequence pooling.
	representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
	
	''' 
	attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
	weighted_representation = tf.matmul(
		attention_weights, representation, transpose_a=True
	)
	weighted_representation = tf.squeeze(weighted_representation, -2)
	'''
	return representation


def construct_model(input_shape):
	'''
	
	construct_model: construct the EQpolarity model
	
	'''
	inputs = layers.Input(shape=input_shape,name='input')

	featuresP = create_cct_model1(inputs)
	featuresP = layers.Flatten()(featuresP)
	featuresP = layers.Dropout(0.2)(featuresP)
	logitp = layers.Dense(1, activation='sigmoid')(featuresP)


	#logitp  = Conv2D(1,  3, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_P')(featuresP)


	model = Model(inputs=[inputs], outputs=[logitp])
# 	model.summary()
	
	return model
