import tensorflow as tf
import pickle
import time
import os
model_file = [file  for file  in  os.listdir() if ".h5" in file][0]
print("model: ",model_file)
model = tf.keras.models.load_model(model_file)
with open("char_dict.p","rb") as f :
    char_dict = pickle.load(f)
with open("text.txt","r",encoding="utf-8") as f :
    text = f.read()

print("Word segmenting...")
s= time.time()
x=[[char_dict.get(char,char_dict["unk"])  for char in text]]  
tag = model.predict(x).argmax(-1).squeeze()
split = []
ni=0
while ni<len(tag):
    i = tag[ni]
    if i == 0 or i==3: 
        split.append((ni,ni))
        ni+=1
        continue
    for nj,j in enumerate(tag[ni+1:]):
        if j==2 : continue
        if i==1 and j==0 :
            split.append((ni,ni+nj))
            ni+=nj+1
            break
            
        if i==1 and j==1 :
            split.append((ni,ni+nj))
            ni+=nj+1
            break
            
        if i==1 and j==3 : 
            split.append((ni,ni+nj+1))
            ni+=nj+1+1
            break
            
        if i==2 and j==0:
            split.append((ni,nj))
            ni+=nj+1
            break
            
        if i==2 and j==1 :
            split.append((ni,nj))
            ni+=nj+1
            break
            
        if i== 2 and j==3 : 
            split.append((ni,ni+nj+1))
            ni+=nj+1+1
            break

text_seg=""
for sep in split:
    start = sep[0]
    end = sep[1]
    if start == end:
        text_seg += text[start]+" "
    else:
        text_seg += text[start:(end+1)]+" "
        
e= time.time()
print("Word segmentation done.","\nCost: ","{}s".format(round(e-s,2)))
with open("text_seg.txt","w",encoding="utf-8") as f :
    f.write(text_seg)
print(text_seg[:20]+" ...")