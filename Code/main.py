"""
# -*- coding: utf-8 -*-

#POC Code  :  Audio Classification  :  H2H DATA
 This code is for audio classification.
 Here we have 3 audio classes namely :
 	1. music
 	2. speech
 	3. noise 
This is done in 2 steps mainly.
	1. First we are seperating speech considering speech feature such as more zero crossing etc.
	2. Ater that music and noise will be seperated. 

Data wrangling and balancing will be done later.

*Here padding and data truncation is done to make data algorithm ready. 

#Copyright@ H2H DATA

#The entire prcess occurs in seven stages-
# 1. DATA INGESTION
# 2. DATA ANALYSIS 
# 3. DATA MUNGING
# 4. DATA EXPLORATION
# 5. DATA MODELING
# 6. HYPER-PARAMETERS OPTIMIZATION
# 7. PREDICTION
# 8. VISUAL ANALYSIS
# 9. RESULTS


Used library
1. pandas
2. numpy
3. time
4. sklearn
5. tensorflow
"""

import helper
import model
import numpy as np 
import time



def main():
	########################################## data ingestion ###############################
	"""
	here we are reading data from audio files which are in differnt folders for seperate labels. Lables are folder name itself.
	"""
	'''
	  DATA INGESTION
	-------------------
	'''
	a = time.time()
	#reading files
	music_files,m_folder = helper.get_files('music')
	noise_files,n_folder = helper.get_files('noise')
	speech_files,s_folder = helper.get_files('speech')

	#getting feature 
	music_audio,SR1 = helper.get_audio(music_files[0:2],m_folder)
	noise_audio,SR2 = helper.get_audio(noise_files[0:2],n_folder)
	speech_audio,SR3 = helper.get_audio(speech_files[0:2],s_folder)
	print 'time to complete data ingestion: ', time.time()-a
	####################################### data ingestion  ends ###############################

	####################################### data analysis ######################################
	"""
	plot zcr value for 1 file from each category i.e. music,noise,sppech
	"""
	music = helper.zcr(music_audio[0])
	noise = helper.zcr(noise_audio[0])
	speech = helper.zcr(speech_audio[0])
	helper.plot_all(music[0],noise[0],speech[0])

	####################################### data analysis ends #################################


	####################################### data exploration ###################################
	"""
	Here we are making data for both models.
	For speech, speech_seration all different type of audio is combined.
	For seperating music only music and noise files are combined. As in 2nd model trauining will be done only on these 2 types.
	For seperating music we need to make all data of equal length in order to pass data to algorithm.
	Long music are truncated.
	while padding is done in short length musics.
	"""
	a = time.time()
	combined_all = sum([music_audio,noise_audio,speech_audio],[])
	combined_nn = sum([music_audio,noise_audio],[])
	combined_nn = helper.make_data_proper(combined_nn)

	label_speech  =  sum([[0]*len(music_audio),[0]*len(noise_audio),[1]*len(speech_audio)],[])
	label = sum([[0]*len(music_audio),[1]*len(noise_audio),[2]*len(speech_audio)],[])
	label_nn = sum([[0]*len(music_audio),[1]*len(noise_audio)],[])
	combined_nn_ = []
	r = len(combined_nn)
	c = len(combined_nn[0])
	i = 0
	for array in combined_nn:
		combined_nn.append(array)

	combined_nn = np.reshape(combined_nn_,(r,c))
	print 'time to complete data exploration: ',time.time() - a
	########################################## data exploration ends #############################


	########################################## modeling ##########################################
	"""
	From 1st model, we are seperating speech file.
	2nd mmodel is uded to classicify mussic and noisse.

	"""


	a = time.time()
	#1st model call 
	speech_index = model.seperate_speech(combined_all)
	speech_count = len(speech_audio)
	speech_predicted = [0]*len(speech_audio)
	for index in speech_index:
		speech_predicted[index] = 1
	print helper.acc(label_speech, speech_predicted) 

	exit()
	#2nd model call
	print model.seperate_music1(combined_nn,label_nn,combined_nn,label_nn)
	print model.seperate_music2(combined_nn,label_nn,combined_nn,label_nn)

	########################################## modeling ends ######################################

if __name__ == '__main__':
	main()

