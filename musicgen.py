from __future__ import print_function
from random import randint

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import scipy.io.wavfile as wav
import wave
import pyaudio
import itertools
from tempfile import TemporaryFile
from collections import Counter

from keras import backend as K
K.clear_session()
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import load_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

from flask import Flask, jsonify, request
from flask_cors import CORS
from flair.models import TextClassifier
from flair.data import Sentence
from flask import session

app = Flask(__name__)
app.secret_key = "super_secret_key"

CORS(app)
#gen_model = GenModel.load_from_file('~/code/models/soundmodel.k')

import nnet


def pad(array, reference, offsets):
	"""
	array: Array to be padded
	reference: Reference array with the desired shape
	offsets: list of offsets (number of elements must be equal to the dimension of the array)
	"""
	# Create an array of zeros with the reference shape
	result = np.zeros(reference.shape)
	# Create a list of slices from offset to offset + shape in each dimension
	insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(a.ndim)]
	# Insert the array in the result at the specified offsets
	result[insertHere] = array
	return result

def wav_to_np(filename):
	data = wav.read(filename)
	np_music = data[1].astype('float32') / 32767.0
	return np_music, data[0]

def np_to_sample(music, block_size=2048):
	blocks = []
	total_samples = music.shape[0]
	num_samples = 0
	print(music.shape)

	rem = music.shape[0] % block_size
	floor = int(np.floor(music.shape[0] / block_size))

	musiclist = list(music)
	pad = [0] * (block_size - rem)
	mus = musiclist + pad
	
	blocks = np.asarray(mus)
	blocks = blocks.reshape((floor+1,block_size))
	print(blocks.shape,'blocks')
	blocks = list(blocks)
	print(len(blocks),len(blocks[0]))


	# while num_samples < total_samples:
	# 	block = music[num_samples:num_samples+block_size]
	# 	if(block.shape[0] < block_size):
	# 		print('oy', block.shape)
	# 		padding = np.zeros((block_size - block.shape[0]))
	# 		block = np.concatenate((block, padding))
	# 	blocks.append(block)
	# 	num_samples += block_size
	# print(len(blocks),len(blocks[0]), 448*2700)

	return blocks

def serialize_corpus(x_train, y_train, seq_len=215):
	seqs_x = []
	seqs_y = []
	cur_seq = 0
	total_seq = len(x_train)
	print('total seq: ', total_seq)
	print('max seq: ', seq_len)

	x = np.asarray(x_train)
	y = np.asarray(y_train)

	while cur_seq + seq_len < total_seq:
		seqs_x.append(x_train[cur_seq:cur_seq+seq_len])
		seqs_y.append(y_train[cur_seq:cur_seq+seq_len])
		cur_seq += seq_len
	print(len(seqs_x),len(seqs_x[0]),len(seqs_x[0][0]))

	return seqs_x, seqs_y

def make_tensors(file, seq_len=215, block_size=2048, out_file='train'):
	'''Have it handle directories *********'''
	music, rate = wav_to_np(file)
	try:
		music = music.sum(axis=1)/2
	except:
		pass

	x_t = np_to_sample(music, block_size)
	y_t = x_t[1:]
	y_t.append(np.zeros(block_size))
	seqs_x, seqs_y = serialize_corpus(x_t, y_t, seq_len)

	nb_examples = len(seqs_x)

	print('\nCalculating mean and variance and saving data\n')
	x_data = np.array(seqs_x)
	y_data = np.array(seqs_y)


	x_data = seqs_x # to be fixed
	y_data = seqs_y
	for examples in range(nb_examples):
		for seqs in range(seq_len):
			for blocks in range(block_size):
				x_data[examples][seqs][blocks] = seqs_x[examples][seqs][blocks]
				y_data[examples][seqs][blocks] = seqs_y[examples][seqs][blocks]
		print('Saved example ', (examples+1), 'of', nb_examples)
	
	mean_x = np.mean(np.mean(x_data, axis=0), axis=0) #Mean across num examples and num timesteps
	std_x = np.sqrt(np.mean(np.mean(np.abs(x_data-mean_x)**2, axis=0), axis=0)) # STD across num examples and num timesteps
	std_x = np.maximum(1.0e-8, std_x) #Clamp variance if too tiny
	print('mean:', mean_x, '\n', 'std:', std_x)

	x_data[:][:] -= mean_x #Mean 0
	x_data[:][:] /= std_x #Variance 1
	y_data[:][:] -= mean_x #Mean 0
	y_data[:][:] /= std_x #Variance 1

	x_data = np.asarray(x_data)
	y_data = np.asarray(y_data)

	np.save(out_file+'_mean', mean_x)
	np.save(out_file+'_var', std_x)
	np.save(out_file+'_x', x_data)
	np.save(out_file+'_y', y_data)
	print('Done!')

	# for x in range(2):
	# 	print(x_data[x], '\n')
	# for x in range(2):
	# 	print(y_data[x], '\n')

	print('mean/std shape: ', mean_x.shape, '\n', std_x.shape)
	return x_data, y_data

def build_model(x_data, y_data, nb_epochs=1, seq_len=215, block_size=2048):
	#input_shape = (seq_len, block_size)
	learning_rate=0.01
	num_epochs = 1
	batch_size = 2
	#lstm = torch.nn.LSTM(input_size=block_size, hidden_size=block_size)
	print(x_data.shape,'xshape')
	print(y_data.shape, 'yyy')
	
	x_data = np.swapaxes(x_data, 1, 0)
	y_data = np.swapaxes(y_data, 1,0)

	print(x_data.shape, type(x_data),'\noopoppo\n')

	dims = x_data.shape
	#exit()
	mylstm = nnet.LSTM(dims[2], 32, batch_size)

	loss_fn = torch.nn.MSELoss(size_average=False)
	optimiser = torch.optim.Adam(mylstm.parameters(), lr=learning_rate)

	#####################
	# Train model
	#####################

	hist = np.zeros(num_epochs)
	x_data = torch.tensor(x_data)
	y_data = torch.tensor(y_data)

	for t in range(num_epochs):
		# Clear stored gradient
		mylstm.zero_grad()
		
		# Initialise hidden state
		# Don't do this if you want your LSTM to be stateful
		mylstm.hidden = mylstm.init_hidden()
		
		# Forward pass
		y_pred = mylstm(x_data)
		print(type(y_pred), type(y_data))
		exit()
		loss = loss_fn(y_pred, y_data)
		if t % 100 == 0:
			print("Epoch ", t, "MSE: ", loss.item())
		hist[t] = loss.item()

		# Zero out gradient, else they will accumulate between epochs
		optimiser.zero_grad()

		# Backward pass
		loss.backward()

		# Update parameters
		optimiser.step()

	return mylstm

def make_brain(timestep=215, block_size=2048):
	print('adding layers...\n')
	model = Sequential()
	model.add(LSTM(block_size, input_shape=(timestep, block_size), return_sequences=True))
	#model.add(Dropout(0.2))
	model.add(Dense(block_size))
	#model.add(Activation('linear'))
	return model

def train_brain(model, x_data, y_data, nb_epochs=1):
	print('training...\n')
	optimizer = RMSprop(lr=0.01)
	model.compile(loss='mse', optimizer='rmsprop')
	model.fit(np.asarray(x_data), np.asarray(y_data), batch_size=500, epochs=nb_epochs, verbose=2)
	#Make it save weights
	print('tttt\n\n\n')
	return model


def run():
	out_file = 'train'
	'''					sample rate * clip len / seq_len '''
	block_size = 2700	# Around min # of samples for human to (begin to) percieve a tone at 16Hz
	seq_len = 215


	'''*****(pseudo-code)*****
	corpus = []
	for file in dir:
		if file.endswith(.wav):
			music, rate = wav_to_np(file)
			music = music.sum(axis=1)/2
			corpus.extend(music)'''
			
	x_data, y_data = make_tensors('./ChillingMusic.wav', seq_len, block_size)


	model = make_brain(seq_len, block_size)
	model = train_brain(model, x_data, y_data)
	masterpiece = compose(model, x_data)

	
	masterpiece = convert_sample_blocks_to_np_audio(masterpiece[0]) #Not final, but works for now
	#print(masterpiece) #			Should now be a flat list
	masterpiece = write_np_as_wav(masterpiece)
	play_music() # Seems to get stuck here (at least sometimes). Need some fix for this. I don't remember if the gui version has that problem...
	print('\n\nWas it a masterpiece (or at least an improvement)?')

	'''Add CNN classifier after converting from Keras to Tensorflow to use generative-adversarial model.
	'''

	return


def load_model():
	pass

def get_seed(seed_len, data_train):
	nb_examples, seq_len = data_train.shape[0], data_train.shape[1]
	r = np.random.randint(data_train.shape[0])
	seed = np.concatenate(tuple([data_train[r+i] for i in range(seed_len)]), axis=0)
	#1 example by (# of examples) timesteps by (# of timesteps) frequencies
	inspiration = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))
	return inspiration

def compose(model, x_data):
	'''Could add choice of length of composition (roughly)'''
	print('composing...\n')
	generation = []
	muse = get_seed(1, x_data)
	for ind in range(1):
		preds = model.predict(muse)
		print(preds)
		print(len(preds), len(preds[0]), len(preds[0][0]))
		generation.extend(preds)
	return generation

@app.route('/api/getaudio', methods=['POST'])
def predict_from_upload():
	block_size = 2700
	seq_len = 215
	filepath = os.join('./upload', request.get_json()['filename']) # path where the file holding seed is located (./uploads/filename)
	# would os.join() work here? not sure if its different bc browser or something
	x_data, y_data = make_tensors(filepath, seq_len, block_size)

	#model = tf.keras.models.load_model('~/code/models/soundmodel.k') this line breaks my laptop but
	#return dummy for the moment


	# how to use flask w vue
	# like how do I call this function from the button in my simpleupload.vue
	# and create a file from here (audio) to be played back in browser

	# secondary - id like to add a file drop zone for my front end to look good lol

	# ok, npm install axios. yea i wasnt sure if wanted user to be able to download the output. idk if thats easy to do
	#
	
	# 1st prob => You have to use axios(library) to make post request from vue to <url>/api/getaudio 
	# to upload and process the uploaded file

	# yes, but there's another library vue-axios which you should not use
	# Regular axios works like a charm

	# for streaming the audio back to browser, are you okay with creating the audio file in a temp folder available 
	# to user?
	# putting the file in temp dir and send the url as response back to vue is easy
	# Other methods include using websocket to live stream the audio to vue as process generates it(may buffer if 
	# server gets slow due to heavy users)
	# Second method is using webrtc, doing it in temp file takes away the pain :D
	# okay!
	# regarding the drop zone, there are articles on vue to create that 
	# you can refer to one of them. Okay, I will start setting up the environment.
	# Need to do a test run on my system before I start working on it :)
	

	
	#masterpiece = compose(model, x_data)
	session['my_result'] = masterpiece # so i need to make a temp file to hold masterpiece to be played back



@app.route('/api/not sure if neededyetlol', methods=['GET'])
def get_result():
	pass

if __name__ == '__main__':
	#run()
	#m = make_brain()
	#optimizer = RMSprop(lr=0.01)
	#m.compile(loss='mse', optimizer='rmsprop')
	print('l')
	model = tf.keras.models.load_model('soundmodel.k')
	#model = load_model('soundmodel.k')
	print(model)


	#x_data, y_data = make_tensors('./ChillingMusic.wav', 215, 2700)
	#build_model(x_data, y_data)
