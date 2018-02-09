import os
import csv
import glob
import numpy
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score as accuracy

#function to get file name
def get_files(ftype):
	"""
	return audio file name and respective folder name
	"""
	folder = os.listdir('../Data/'+ftype)
	folder = '../Data/'+ftype +'/'+folder[0]
	files = os.listdir(folder)
	audio_files = filter(lambda x: x.split('.')[-1] == 'wav', files)
	return audio_files,folder



#function to read audio file
def get_audio(files, folder):
	"""
	returns audio and sampling ratio
	"""
	Y = []
	SR = []
	if files:
		for file in files:
			try:
				y,sr = librosa.load(folder+'/'+file)
			except:
				print 'can not read given file'
				continue
			Y.append(y)
			SR.append(sr)
	return Y,SR



#function to trancate long audio file to input to rnn
def trancate(length,allowed_length,audio):
	"""
	if audio file is too long then remove starting and ending of audio
	from start remove extra then mean-2*std by 2
	from end remove mean +2*std by 2
	"""
	a = length
	b = allowed_length
	if a > b:
		remove = b -a 
		if remove%2 == 0:
			audio= audio[remove/2 : a-remove/2]
		else:
			audio= audio[int(remove/2) : a-int(remove/2) -1]
	return audio



#function to pad short audio file to input to rnn
def pad(length,required_length,audio):
	"""
	if audio file is too short then i pads starting and ending of audio
	from start it pads less then mean-2*std by 2
	from end it pads mean +2*std by 2
	"""
	a = length
	b = required_length
	if a < b:
		add = b -a 
		if add%2 == 0:
			audio= [0]*add/2 + list(audio) + [0]*add/2
		else:
			audio= [0]*add/2 + list(audio) + [0]*(add/2+1)
		return np.array(audio)
	else:
		return audio


#function to make data rnn ready
def make_data_proper(audios):
	"""
	returns audio file after padding and turncating required audio to get proper size array
	"""
	length_list = np.array(map(lambda x: len(x), audios))
	l_mean =length_list.mean()
	l_min = length_list.min()
	l_max = length_list.max()
	l_std = length_list.std()
	for i, audio in enumerate(audios):
		audio = trancate(length_list[i], l_mean + 2*l_std, audio)
		audio = pad(length_list[i], l_mean - 2*l_std, audio)
		audios[i] = audio
	return audios


#function to find zcr value
def zcr(audios):
	"""
	returns zcr value for an audio file
	"""
	return librosa.feature.zero_crossing_rate(y = np.array(audios))

#function to get accuracy
def acc(actual,predicted):
	return accuracy(actual,predicted)

#function to plot data
def plot_all(a,b,c):
	'''

	'''
	plt.figure()
	plt.subplot(311)
	plt.plot(a)
	plt.ylabel('music')
	plt.subplot(312)
	plt.plot(b)
	plt.ylabel('noise')
	plt.subplot(313)
	plt.plot(c)
	plt.ylabel('speech')
	plt.show()