## Import
import numpy as np 
import pandas as pd 
# import matplotlib.pyplot as plt 
# import plotly.express as px
import tensorflow as tf
import config
import os
import sys
import time
import datetime
import pickle
import shutil
##########load config########################
model_name = config.model_name
assert model_name in ["BiGRU", "BiLSTM"] , "model_name only given BiGRU, BiLSTM"
dataset = config.dataset
assert dataset in ["msr", "pku"] , "Dataset Only Given msr,pku"
embedding_dim = config.embedding_dim
drop_rate = config.drop_rate 
rnn_dim = config.rnn_dim

# compile
learning_rate = config.learning_rate
beta_1 = config.beta_1
beta_2 = config.beta_2

# fit
batch_size = config.batch_size
epochs = config.epochs
workers = config.workers

############################################

## create a folder
now = datetime.datetime.now()
date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
dirName = '{}-{}_{}'.format(model_name,dataset,date_time)
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")

## Copy Config.py 
shutil.copy2('config.py', "./{}/".format(dirName) +'config.py')
## Copy Segmenter.py 
shutil.copy2('Segmenter.py', "./{}/".format(dirName) +'Segmenter.py')
## Copy text.txt 
shutil.copy2('text.txt', "./{}/".format(dirName) +'text.txt')

## Laod msr data
with open("./icwb2-data/training/{}_training.utf8".format(dataset),encoding="utf-8") as f:
    msr_training=f.readlines()
print("How many lines in {}_training.utf8  ? ".format(dataset),len(msr_training))

## Tagging
msr_training_tagged=[]
for line in msr_training:
    word_list = line.strip().split()
    tag=[]
    for word in word_list:
        if len(word) == 1:
            tag.append("S")
        else:
            tag.append("B"+"M"*len(word[1:-1])+"E")
    msr_training_tagged.append((word_list,tag))
# print(msr_training_tagged[0][0][:])
# print(msr_training_tagged[0][1][:])


## Making Word and Char Dictionary
word_count = dict()
for text, tag in msr_training_tagged:
    for word in text :
        if word not in word_count:
            word_count[word] = 1
        else :
            word_count[word]+=1
# pd.Series(word_count).sort_values(ascending=False)

char_count={}
for text, tag in msr_training_tagged:
    for char in "".join(text):
        if char not in char_count: char_count[char]=1
        else: char_count[char]+=1
char_dict = {} 
char_dict["pad"]=0
char_dict["unk"]=1
for i, char in enumerate(char_count.keys()):
    char_dict[char] = i+2 
char_size=len(char_dict )
print("char_size: ",char_size)
with open("./{}/".format(dirName)+"char_dict.p","wb") as f :
    pickle.dump(char_dict,f)
###### Preparing Traina and Val Dataset####
print("Preparing Traina and Val Dataset...")
longest=0
x_train=[]
for text,tag in msr_training_tagged:
    temp=[ char_dict[char] for char in "".join(text)]
    if len(temp) > longest : longest = len(temp)

print("longest text: ", longest)
for text,tag in msr_training_tagged:
    temp=[ char_dict[char] for char in "".join(text)]
    if len(temp) < longest : temp+=[0]*(longest-len(temp))
    x_train.append(temp)
x_train = np.array(x_train)
# x_train.shape

tag_map={"S":[1,0,0,0],"B":[0,1,0,0],"M":[0,0,1,0],"E":[0,0,0,1]}
y_train=[]
for text,tag in msr_training_tagged:
    temp=[ tag_map[tag] for tag in "".join(tag)]
    if len(temp) < longest : temp += [[0,0,0,0]]*(longest-len(temp))
    y_train.append(temp)
y_train = np.array(y_train)
# y_train.shape    

x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]
if config.is_test_mode :
    x_train = x_train[:5]
    y_train = y_train[:5]
    x_val = x_val[:5]
    y_val = y_val[:5]

#############################################

############# Model-Building  #################
print("Model-Buildingt...")
tf.keras.backend.clear_session()
input_ = tf.keras.layers.Input((None,))
output_ = tf.keras.layers.Embedding(input_dim=char_size, output_dim=embedding_dim, mask_zero=True)(input_)

if model_name == "BiGRU" :
    output_ = tf.keras.layers.Bidirectional(
                            tf.keras.layers.GRU(rnn_dim, return_sequences=True), 
                            input_shape=(longest, embedding_dim))(output_)
if model_name == "BiLSTM" :
    output_ = tf.keras.layers.Bidirectional(
                            tf.keras.layers.LSTM(rnn_dim, return_sequences=True), 
                            input_shape=(longest, embedding_dim))(output_)

output_ = tf.keras.layers.Dropout(drop_rate)(output_)
output_ = tf.keras.layers.Dense(4,activation="sigmoid")(output_)
model = tf.keras.Model(input_, output_)

# Open the file
with open("./{}/".format(dirName) + 'model_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
model.summary()
#############################################

################### Compile #################
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1,
                                     beta_2=beta_2, amsgrad=False)
model.compile(optimizer, 'categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
#############################################

################# Training ################## 
print('Train...')
History = model.fit(
                x=x_train, y=y_train, 
                validation_data=(x_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                workers=workers,
                )
model.reset_metrics()
model.save("./{}/".format(dirName)+"{}.h5".format(model_name))

## save result of training
print("Save result of training...")
df = pd.DataFrame(History.history)
df.to_csv("./{}/".format(dirName)+"history.csv")

ax = pd.DataFrame(History.history)[["loss","val_loss"]].plot()
fig = ax.get_figure()
fig.savefig("./{}/".format(dirName)+"loss.png")

ax = pd.DataFrame(History.history)[["categorical_accuracy","val_categorical_accuracy"]].plot()
fig = ax.get_figure()
fig.savefig("./{}/".format(dirName)+"categorical_accuracy.png")
###############################################

##################### Test ####################
print("Testing...")
## Load test file 
with open("./icwb2-data/gold/{}_test_gold.utf8".format(dataset),encoding="utf-8") as f:
    msr_test=f.readlines()
print("How many lines in {}_test_gold.utf8".format(dataset),len(msr_test))

msr_test_tagged=[]
for line in msr_test:
    word_list = line.strip().split()
    tag=[]
    for word in word_list:
        if len(word) == 1:
            tag.append("S")
        else:
            tag.append("B"+"M"*len(word[1:-1])+"E")
    msr_test_tagged.append((word_list,tag))
# print(msr_test_tagged[5][0][:])
# print(msr_test_tagged[5][1][:])

## preparing dataset to calculate metrics
x_test=[]

#### time it 1
s=time.time()
for text,tag in msr_test_tagged:
    temp=[ char_dict[char] if char in char_dict else char_dict["unk"] for char in "".join(text)]
    if len(temp) < longest : temp+=[0]*(longest-len(temp))
    x_test.append(temp)
x_test = np.array(x_test)
e=time.time()
p1=e-s
####
tag_map={"S":[1,0,0,0],"B":[0,1,0,0],"M":[0,0,1,0],"E":[0,0,0,1]}
y_test=[]
for text,tag in msr_test_tagged:
    temp=[ tag_map[tag] for tag in "".join(tag)]
    if len(temp) < longest : temp += [[0,0,0,0]]*(longest-len(temp))
    y_test.append(temp)
y_test = np.array(y_test)

text_len=[]
for text,tag in msr_test_tagged:
    text_len.append(len("".join(tag)))
#### time it 2
s = time.time()
y_predict = model.predict(x_test)
e = time.time()
p2 = e-s
####
print("testset-predict cost: ",p1+p2)

class word_segment_metrics:
    def __init__(self,y_predict,y_true,text_len):
        self.y_predict_flat, self.y_true_flat = self.flatten_and_truncat(y_predict,y_true,text_len)
        self.prf = self.prf_calculator()
        
    def flatten_and_truncat(self,y_predict,y_true,text_len):
        y_predict_flat = []
        y_true_flat = []
        for row_predict,row_true,size in zip(y_predict.argmax(-1),y_true.argmax(-1),text_len):
            y_predict_flat+=row_predict[:size].tolist()
            y_true_flat+=row_true[:size].tolist()   
        return np.array(y_predict_flat), np.array(y_true_flat)
    def prf_calculator(self):
        p_pred_S=len(self.y_predict_flat[self.y_predict_flat==0])
        p_true_S = len(self.y_true_flat[self.y_true_flat==0])
        tp_S = len(self.y_predict_flat[(self.y_predict_flat==0) *(self.y_true_flat==0)])
        P_S = (tp_S/p_pred_S )if p_pred_S != 0 else None
        R_S = (tp_S/p_true_S) if p_true_S != 0 else None
        try :
            F_S = 2*((P_S*R_S)/(P_S+R_S)) 
        except:
            F_S = None

        p_pred_B=len(self.y_predict_flat[self.y_predict_flat==1])
        p_true_B = len(self.y_true_flat[self.y_true_flat==1])
        tp_B = len(self.y_predict_flat[(self.y_predict_flat==1) *(self.y_true_flat==1)])
        P_B=(tp_B/p_pred_B )if  p_pred_B != 0 else None
        R_B=(tp_B/p_true_B) if  p_true_B != 0 else None
        try :
            F_B = 2*((P_B*R_B)/(P_B+R_B)) 
        except:
            F_B = None

        p_pred_M=len(self.y_predict_flat[self.y_predict_flat==2])
        p_true_M = len(self.y_true_flat[self.y_true_flat==2])
        tp_M = len(self.y_predict_flat[(self.y_predict_flat==2) *(self.y_true_flat==2)])
        P_M = (tp_M/p_pred_M) if  p_pred_M != 0 else None
        R_M = (tp_M/p_true_M) if  p_true_M != 0 else None
        # F_M = 2*((P_M*R_M)/(P_M+R_M)) if  (P_M+R_M) == 0 else None
        try :
            F_M = 2*((P_M*R_M)/(P_M+R_M)) 
        except:
            F_M = None

        p_pred_E=len(self.y_predict_flat[self.y_predict_flat==3])
        p_true_E = len(self.y_true_flat[self.y_true_flat==3])  
        tp_E = len(self.y_predict_flat[(self.y_predict_flat==3) *(self.y_true_flat==3)])
        P_E= (tp_E/p_pred_E) if  p_pred_E != 0 else None
        R_E= (tp_E/p_true_E) if  p_true_E != 0 else None
        try :
            F_E = 2*((P_E*R_E)/(P_E+R_E))
        except:
            F_E = None

        return {
            "P":[P_S,P_B,P_M,P_E],
            "R":[R_S,R_B,R_M,R_E],
            "F":[F_S,F_B,F_M,F_E],
               }
ws_metrics=word_segment_metrics(y_predict,y_test,text_len)
df = pd.DataFrame(ws_metrics.prf, index = ["S", "B", "M", "E"])
print(df)
df.to_csv("./{}/".format(dirName)+"metrics.csv")
print("Tese Done.")
###############################################################