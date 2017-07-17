"""
0:Requirements
"""
MeCab(install by Brew)
natto.py
gensim
pandas
csv
sklearn

"""
1:import
"""
from corpaspredictor import Dict2vec,randomforestmodelinitialize,randomforesttrain,predict

"""
2:varience is fixed
"""
datapath = '.' #
dictionary_save_path = '.' #
no_below = 5 #gensim_dictionary_parameta
no_above = 0.05 #gensim_dictionary_parameta
keep_n = 1000 #gensim_dictionary_parameta
data_number = 2 #

"""
3:create object and create dictionaly and corpas
"""

obj = Dict2vec()

dense,train_label,dictionaly = obj.create_dict(datapath=datapath, 
		dictionary_save_path=dictionary_save_path, 
		no_below=no_below, 
		no_above=no_above, 
		keep_n=keep_n, 
		data_number=data_number)

b = a.load_dict(dictionary_save_path=dictionary_save_path)

"""
4:estimator initialize
"""

estimator = randomforestmodelinitialize()

"""
5:estimator train
"""
randomforesttrain(dense=dense, train_label=train_label, dictionary=dictionaly, estimator=estimator)

"""
6:predict
"""
analisys_words = '今日は焼鮭定食がたべたい。笑顔。めちゃくちゃあっさりしていて、塩加減のちょうどいいものが希望。副菜もいくつかあるといいなあ。野菜多めも◎'
predict(analisys_words=analisys_words, dictionary=dictionaly, estimator=estimator)