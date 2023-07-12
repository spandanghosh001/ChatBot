from flask import Flask, render_template, request
import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

f = open('./data/datascience.txt','r',errors='ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower() #Converting entire text to lowercase
nltk.download('punkt') #Using the Punkt Tokenizer
nltk.download('wordnet') #using the wordnet dictionary
nltk.download('omw-1.4')

sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

greet_inputs = ('hello','hi','whassup','how are you?')
greet_responses = ('hi','Hey','Hey There!')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

def response(user_response):
    TfidVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words = 'english')
    tfidf = TfidVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        return "I am sorry. Unable to understand you!"
    else:
        return sentence_tokens[idx]
    

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    user_response = request.args.get('msg')
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thank you' or user_response == 'thanks'):
            return str('You are welcome..')
        else:
            if(greet(user_response) != None):
                return str(greet(user_response))
            else:
                sentence_tokens.append(user_response)
                return response(user_response)
    else:
        return str('Goodbye!')



if __name__ == "__main__":
    app.run()
