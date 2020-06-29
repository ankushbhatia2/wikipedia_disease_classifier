#IMPORTS 
from tensorflow.keras.models import load_model
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pprint import pprint


import os, re
import tensorflow.keras.backend as K
import wikipedia
import stop_words
import pickle
import numpy as np
import argparse


def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


#LOAD PRETRAINED MODELS 
lda = LdaModel.load("lda.model")
dictionary = Dictionary.load("corpora.dictionary")
classifier = load_model("classifier.h5", custom_objects={"get_f1":get_f1})


with open("vocab.pkl", 'rb') as f:
    top_k_words = pickle.load(f)

def keep_top_k_words(text):
    return [word for word in text if word in top_k_words]


#For preprocessing/cleaning
remove_ref = re.compile("\[[0-9]\]")
def cleanText(text):
    return remove_ref.sub("", text).lower()

def initial_clean(text):
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case the text
    text = word_tokenize(text)
    return text

stopWords = stop_words.get_stop_words('english')
def remove_stop_words(text):
    return [word for word in text if word not in stopWords]

stemmer = PorterStemmer()
def stem_words(text):
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
    except IndexError: # the word "oed" broke this, so needed try except
        pass
    return text

def apply_all(text):
    return stem_words(remove_stop_words(initial_clean(text)))




#TESTING Pages
def get_info(page_html):
    page_soup = BeautifulSoup(page_html)
    info_dict = {}
    info_class = page_soup.find("table", {"class":"infobox"})
    for tr in info_class.find_all('tr'):
        if tr.find("th") and tr.find("td"):
            info_dict[tr.th.text.strip().lower()] = cleanText(tr.td.text.strip())
    return info_dict

def predict_page(page_name):
    page = wikipedia.page(page_name)
    test_toks = keep_top_k_words(apply_all(page.content))
    test_bow = dictionary.doc2bow(test_toks)
    test_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=test_bow)])
    pred = classifier.predict(np.array([test_doc_distribution]))
    is_disease = pred[0][0] > 0.9
    info_dict = {}
    info_dict['is disease'] = is_disease
    if is_disease:
        info_dict['disease name'] = page.title
        info_dict.update(get_info(page.html()))
    return info_dict


def main():
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group("required arguments")

    # required arguments
    required.add_argument("--page", required=True, help="Name of wikipedia page")

    args = parser.parse_args()
    info_dict = predict_page(args.page)
    
    pprint(info_dict)
if __name__ == '__main__':
    main()
