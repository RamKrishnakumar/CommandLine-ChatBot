import random
import json 
import pickle 
import numpy as np 

import nltk 
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# function to clean up sentence and to tokenize and lemmatize
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# function to convert words into bag of sentence 
# into the list of 0's & 1's to indicate wheather word is there or not
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
      for i, word in enumerate(words):
          if word == w:
              bag[i] = 1
    return np.array(bag)

# function to predict the sentence
def predict_class(sentence):
    bow = bag_of_words(sentence) # bow ~ bag of words
    res = model.predict(np.array([bow]))[0] # [bow] = to pass as a list, [0] index
    ERROR_THRESHOLD = 0.25 # allow to stricting uncertainity
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] #[i,r] ~ [index, result] function runs only if res > threshold
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list =[]
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Go! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
