import csv,pickle,json,numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
texts=[]; labels=[]
with open("dataset.csv",encoding="utf-8") as f:
 for r in csv.DictReader(f): texts.append(r["text"]); labels.append(r["sentiment"])
tok=Tokenizer(num_words=3000,oov_token="<OOV>"); tok.fit_on_texts(texts)
X=pad_sequences(tok.texts_to_sequences(texts),maxlen=30,padding="post")
enc=LabelEncoder(); y=np.eye(3)[enc.fit_transform(labels)]
model=Sequential([Embedding(3000,64,input_length=30),LSTM(64),Dropout(.3),Dense(32,activation="relu"),Dense(3,activation="softmax")])
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(X,y,epochs=35,batch_size=4,verbose=1)
model.save("lstm_sentiment_model.keras")
pickle.dump(tok,open("tokenizer.pkl","wb")); json.dump(enc.classes_.tolist(),open("label_classes.json","w"))
print("LSTM model saved successfully.")
