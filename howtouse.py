"""
0-1:Requirements package
"""
Brewで入れるもの（MacOSの場合）
MeCab(install by Brew)
（brew install mecab-ipadic　がオススメ　MeCab用辞書と同時に勝手にMecabもインストールされる。）

$ brew install mecab-ipadic
$ mecab -v #インストール完了しているか確認

pipで入るもの（reqirements.txt整備予定）
natto.py
gensim
pandas
csv
sklearn

"""
0-2:Requirements data 

data path of raw data #here you put data
"sentences.csv" rawdata to make dictionary
"data0.csv, data1.csv, data2.csv,…"　raw_data and answer_label to make dense

とりあえず、
データパス変数にデータパス（相対・絶対両方可）を記しますが、そのときに次の2つの名前のファイルをセッティングしてください
１："sentences.csv" 辞書をつくるときの素材です。
２："data0.csv, data1.csv, data2.csv,…"　素性ベクトルをつくるときの素材です。文章とその正解ラベル
										（デモでは料理国別に0,1の２つでラベルを張っている複数になっても計算可）が書かれている。
"""

"""
1:import
"""
from densenlpt import Dict2dense,Predictor

"""
2:varience is fixed
"""
datapath = '.' #data path of raw data #here you put data "sentences.csv" and "data0.csv, data1.csv, data2.csv,…"　please look demodata.
dictionary_save_path = '.' #save path
no_below = 5 #gensim_dictionary_parameta
no_above = 0.05 #gensim_dictionary_parameta
keep_n = 1000 #gensim_dictionary_parameta
data_number = 2 #number of data in datapath
dict_filter=False #if number of words is low, then False may be work.

"""
3:create object and create dictionaly and corpas
"""
obj = Dict2dense(datapath=datapath, 
		dictionary_save_path=dictionary_save_path, 
		no_below=no_below, 
		no_above=no_above, 
		keep_n=keep_n, 
		data_number=data_number,
		dict_filter=dict_filter)

dense,train_label,dictionary = obj.create_dict()

saved_dict = obj.load_dict() #load dictionary

"""
4:estimator initialize
"""
modelobj = Predictor(dense,train_label,dictionary,None)
modelobj.randomforestmodel_initial_train()
modelobj.randomforestmodel_retrain(dense,train_label,dictionary,modelobj.estimator)

"""
5:estimator prediction
"""
words = '今日は焼鮭定食がたべたい。笑顔。めちゃくちゃあっさりしていて、塩加減のちょうどいいものが希望。副菜もいくつかあるといいなあ。野菜多めも◎'
answer = modelobj.onebyonepredict(words)
print(answer[0])



