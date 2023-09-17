from keras.preprocessing.text import Tokenizer, tokenizer_from_json
import re
import nltk
import json
from nltk.corpus import stopwords
from keras.utils import pad_sequences
from keras.layers import Conv1D, Flatten, Embedding, LSTM, Dropout
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from tensorflow.keras.models import load_model

maxlen = 100
size = 100
stopwords_list = set(stopwords.words('english'))
vocab_length = 90783

def preprocess(text):
    sentence = re.sub('[^a-zA-Z]', ' ', text.lower())
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub('<.*?>', '', sentence) # HTML tags
    sentence = re.sub(r'\d+', '', sentence) # numbers
    sentence = re.sub(r'[^\w\s]', '', sentence) # special characters
    sentence = re.sub(r'http\S+', '', sentence) # URLs or web links
    sentence = re.sub(r'@\S+', '', sentence) # mentions
    sentence = re.sub(r'#\S+', '', sentence)

    if len(text) > 20:
        pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
        sentence = pattern.sub('', sentence)
    return sentence.strip()

token = open("./models/tokenizer.json", "r")
data = json.load(token)
token.close()

word_tokenizer = tokenizer_from_json(data)
lstm_model = load_model("./models/lstm_model_acc_0.866_imdb.h5")
print(lstm_model.summary())

def sentiment_analysis(text):
    text = preprocess(text)
    text = word_tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, padding='post', maxlen=maxlen)

    answer = lstm_model.predict([text])
    
    return answer

def sentiment_analysis_many(textList):
    for i in range(len(textList)):
        textList[i] = preprocess(textList[i])

    text = word_tokenizer.texts_to_sequences(textList)
    text = pad_sequences(text, padding='post', maxlen=maxlen)

    answer = lstm_model.predict(text)
    
    return answer


if __name__ == "__main__":
    i = input()
    answer = sentiment_analysis(i)
    print(answer)



