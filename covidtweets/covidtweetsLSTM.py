import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import re
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import spacy
import demoji
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
import numpy as np
import contractions
import unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical,pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,SpatialDropout1D,Dense,Bidirectional,Dropout
from sklearn.preprocessing import LabelEncoder



nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
sw = set(stopwords.words("english"))


df = pd.read_csv("C:/nlp_projects/covidtweets/vaccination_all_tweets.csv",delimiter=',')
df.head(10)
df.isnull().sum()

df = df[['text']]
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)


def clean_text(text):

    text = text.lower()


    text = re.sub(r'<.*?>', '', text)


    text = contractions.fix(text)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)


    text = re.sub(r'@\w+|#\w+', '', text)


    text = demoji.replace(text, '')

    text = unicodedata.normalize("NFKD", text)


    text = re.sub(r'[^a-zA-Z\s]', '', text)


    tokens = word_tokenize(text)
    
    
    tokens = [word for word in tokens if word not in sw]

    text = " ".join(tokens).strip()

    return text




def stem_text(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stem_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stem_tokens)



df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].apply(stem_text)


sample_txt=  " ".join(i for i in df['text'])


wc = WordCloud(colormap="Set2",collocations=False).generate(sample_txt)
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.show()


most_common_words = nltk.FreqDist(TextBlob(sample_txt).words).most_common(50)
print(f"Top 50 Most Common Words: {most_common_words}")



nlp = spacy.load("en_core_web_sm")



doc = nlp(sample_txt[:2000])



for token in doc:
    print(token.text,token.pos,token.dep_)


for ent in doc.ents:
    print(ent.text, "|",spacy.explain(ent.label_))


def polarity(text):
    return TextBlob(text).polarity

df['polarity'] = df['text'].apply(polarity)


def sentiment(label):
    if label == 0:
        return "NEUTRAL"
    elif label < 0:
        return "NEGATIVE"
    elif label > 0:
        return "POSITIVE"



df['sentiment'] = df['polarity'].apply(sentiment)


plt.figure(figsize=(10,6))
df['sentiment'].value_counts().plot(kind='bar',rot=0)
plt.show()



le = LabelEncoder()




X = df['text']
y = df['sentiment']
y = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
print("length of word index: ")
print(len(word_index))


X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)




max_length = 0
for sequence in X_train:
    sequence_length = len(sequence)
    if sequence_length > max_length:
        max_length = sequence_length

print("Max Length of Sequences: ",max_length)



y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = pad_sequences(X_train,padding='post')
X_test = pad_sequences(X_test,padding="post")




RNN = Sequential()
RNN.add(Embedding(len(word_index)+1,output_dim=20,input_length=max_length))
RNN.add(SpatialDropout1D(0.3))
RNN.add(Bidirectional(LSTM(10,dropout=0.1,recurrent_dropout=0.1)))
RNN.add(Dropout(0.2))
RNN.add(Dense(20,activation='relu'))
RNN.add(Dropout(0.1))
RNN.add(Dense(3,activation='softmax'))
RNN.summary()


RNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = RNN.fit(X_train,y_train,epochs=10,batch_size=64,validation_split=0.1)
pred = RNN.predict(X_test)
loss,acc = RNN.evaluate(X_test,y_test)
print(f"Testing Loss: {loss*100:.2f}%")
print(f"Testing accuracy: {acc*100:.2f}%")



plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Training and Validation Accuracy")
plt.show()

plt.figure(figsize=(10,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Training and Validation loss")
plt.show()




y_true = np.argmax(y_test,axis=1)
y_pred = np.argmax(pred,axis=1)



f1 = f1_score(y_true, y_pred,average="weighted")
print(f'F1 Score from LSTM: {f1*100:.2f}%')

acc = accuracy_score(y_true, y_pred)
print(f'accuracy: {acc*100:.2f}')


def plot_cofusison_matrix(y_true,y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    return sns.heatmap(conf_matrix,fmt='d',annot=True,cmap='coolwarm',xticklabels=['Negative','Neutral','Positive'],yticklabels=['Negative','Neutral','Positive'])


plot_cofusison_matrix(y_true, y_pred)




