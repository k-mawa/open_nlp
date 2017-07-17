from natto import MeCab
from gensim import corpora, matutils
import pandas as pd
import csv
import os
import sys
import time
from sklearn.ensemble import RandomForestClassifier

nm = MeCab('-Owakati')

datapath = '.'
dictionary_save_path = '.'
no_below=5 #gensim_dictionary_parameta
no_above=0.05 #gensim_dictionary_parameta
keep_n=1000 #gensim_dictionary_parameta
data_number = 2 #リソースの数
dense = []
train_label =[]
dictionary = None
estimator = None
analisys_words = None
"""
1
"""
class Dict2vec:
	def __init__(self):
		pass

	def create_dict(self, 
		datapath=datapath, 
		dictionary_save_path=dictionary_save_path, 
		no_below=no_below, 
		no_above=no_above, 
		keep_n=keep_n, 
		data_number=data_number):

		train_raw_data =[]
		train_label =[]
		for k in range(data_number):

		    csvdata = pd.read_csv(os.path.join(datapath,'data{}.csv'.format(k)), header=None)

		    for i in csvdata[0]:
		        train_raw_data.append(i)
		    for i in csvdata[1]:
		        train_label.append(i)

		raw_word_list =[]

		with open(os.path.join(datapath,'sentences.csv'), 'r') as f:
		    reader = csv.reader(f)
		    header = next(reader)
		    for row in reader:
		        raw_word_list.append(row) 

		wakati_by_sentnces_list = []

		for i in range(len(raw_word_list)):
		    wakati_by_sentnces_list.append([n.surface for n in nm.parse(raw_word_list[i][0], as_nodes=True) if n.is_nor()])
		        
		full_wakati_words = []
		for i in range(len(raw_word_list)):
		    for j in [n.surface for n in nm.parse(raw_word_list[i][0], as_nodes=True) if n.is_nor()]:
		        full_wakati_words.append(j)

		print("number of scentences is ",len(raw_word_list))
		print("number of wakati_gaki",len(full_wakati_words))


		dictionary = corpora.Dictionary(wakati_by_sentnces_list)
		print(dictionary)

		dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
		print(dictionary)

		key_index=[]

		for i in dictionary.token2id.keys():
		    key_index.append(i)

		print("part of new dict contents")
		print("===========")
		print("word：ID")
		for i in range(5):
		    print(key_index[i],":",dictionary.token2id[key_index[i]])
		print("===========")

		dictionary.save_as_text(os.path.join(dictionary_save_path,'dictionary.txt'))
		print("new dict saved at {}".format(os.path.join(dictionary_save_path,'dictionary.txt')))

		dense =[]
		for j in train_raw_data:
			tmp = dictionary.doc2bow(list(j))
			dense.append(list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])) 

		print("length of dense:",len(dense))
		print("length of train_label:",len(train_label))

		return dense, train_label, dictionary


	def load_dict(self, dictionary_save_path=dictionary_save_path):
		return corpora.Dictionary.load_from_text(os.path.join(dictionary_save_path,'dictionary.txt'))

def randomforestmodelinitialize():
	estimator = RandomForestClassifier()
	return estimator

def randomforesttrain(dense=dense, train_label=train_label, dictionary=dictionary, estimator=estimator):
	estimator.fit(dense, train_label)

def predict(analisys_words=analisys_words, dictionary=dictionary, estimator=estimator):
	test_dense = []
	test_tmp = dictionary.doc2bow(list(analisys_words))
	test_dense.append(list(matutils.corpus2dense([test_tmp], num_terms=len(dictionary)).T[0])) #vector by doc2bow corpas
	label_predict = estimator.predict(test_dense[0])
	print("prediction of label:",label_predict[0])
	return label_predict

