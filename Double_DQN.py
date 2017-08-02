# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from keras.models import Model,Sequential
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten,Activation
from keras.optimizers import RMSprop,Adam
from keras.initializations import normal

#import flappybird as game
import wrapped_flappy_bird as game
import skimage
import sys
import random
from skimage import color
from skimage import transform,exposure
from time import gmtime, strftime

discount_factor = 0.99
ep = 0.1
batch_size = 32
rows,cols = 84,84
stack = 4

def initialize(shape,name,dim_ordering=None):
    return normal(shape, scale=0.01, name = name)

def getModel(mode):
	inputs = Input(shape=(rows,cols,stack))
	layer1 = Convolution2D(32, 8, 8, activation='relu',subsample=(4,4),init=initialize, border_mode='same')(inputs)
	layer2 = Convolution2D(64, 4, 4, activation='relu',subsample=(2,2),init=initialize, border_mode='same')(layer1)
	layer3 = Convolution2D(64, 3, 3, activation='relu',subsample=(1,1),init=initialize, border_mode='same')(layer2)
	flatten = Flatten()(layer3)
	layer4 = Dense(512, activation='relu',init=initialize)(flatten)
	output = Dense(2,init=initialize , activation = 'linear')(layer4)
	model = Model(inputs , output)
	rmsprop = RMSprop(lr=0.0025)
	if mode == 'Run':
		model.load_weights("model-ddqn.h5")
	model.compile(loss='mse',optimizer=rmsprop)
	return model

def processImage(img):
	img = skimage.color.rgb2grey(img)
	img = skimage.transform.resize(img,(rows,cols))
	img = skimage.exposure.rescale_intensity(img,out_range=(0,255))
	return img

def trainModel(predict_model,actual_model,mode):
	fb = game.GameState()
	replay_mem = []
	i_0, r_0, isDead = fb.frame_step(0)
	i_0 = processImage(i_0)
	state_0 = np.stack((i_0,i_0,i_0,i_0), axis=2)
	state_0 = state_0.reshape(1,rows,cols,stack)
	state_t = state_0
	t = 0
	if mode == 'Train':
		log = file(strftime("%Y-%m-%d-%H:%M:%S", gmtime()) , 'w')
	while True:
		loss = 0
		q_max = 0
		if random.random() < ep and mode == 'Train':
			print ('taking random action')
			flap = random.randint(0,1)
		else:
			q = predict_model.predict(state_t)
			flap = np.argmax(q[0])
			q_max = max(q[0])
		i_t, r_t, isDead = fb.frame_step(flap)
		i_t = processImage(i_t)
		i_t = i_t.reshape(1, rows, cols ,1)
		state_t1 = np.append(i_t, state_t[:, :, :, :3], axis=3)

		replay_mem += [(state_t,state_t1,flap,r_t,isDead)]
		if t > 50000:
			replay_mem = replay_mem[1:]

		if t > 3000 and mode == 'Train':
			batch = random.sample(replay_mem,batch_size)
			X =  np.zeros((batch_size,rows,cols,4))
			Y = np.zeros((batch_size,2))
			for i in range(len(batch)):
				X[i:i+1] = batch[i][0]
				q = predict_model.predict(batch[i][0])
				Y[i] = q[0]
				action = batch[i][2]
				reward = batch[i][3]
				
				if batch[i][4] == True:
					Y[i,action] = reward
				else:
					max_a = np.argmax(predict_model.predict(batch[i][1])[0])
					q = actual_model.predict(batch[i][1])[0][max_a]
					Y[i,action] = reward + discount_factor*q
			loss = predict_model.train_on_batch(X,Y)
		if t%1000 == 0 and mode == 'Train':
			print ('sync two models...')
			actual_model.set_weights(predict_model.get_weights())
			predict_model.save_weights("model-ddqn.h5", overwrite=True)
			# Do I need to re-compile the model here ?  
		print ('T = '+str(t)+' Loss = '+str(loss)+' q-max = '+str(q_max)+' reward = '+str(r_t)+' action = '+str(flap))
		if mode == 'Train':
			log.write('T = '+str(t)+' Loss = '+str(loss)+' q-max = '+str(q_max)+' reward = '+str(r_t)+' action = '+str(flap) + '\n')

		t += 1
		state_t = state_t1


def initModel(mode):
	predict_model = getModel(mode)
	actual_model = getModel(mode)
	actual_model.set_weights(predict_model.get_weights())
	print (predict_model.summary())
	trainModel(predict_model,actual_model,mode)

if __name__ == "__main__":
	mode = 'Train'
	if len(sys.argv) >= 1:
		mode = sys.argv[1]
	initModel(mode)