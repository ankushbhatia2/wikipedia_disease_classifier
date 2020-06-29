#Inports
import os, re
import wikipedia
import stop_words
import numpy as np


from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec, LdaModel
from nltk import FreqDist
from nltk.stem import PorterStemmer
from gensim import models, corpora, similarities

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K



EXPERIMENT = False


"""
CLEANING
"""
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


"""
LOAD WIKIPEDIA DATA FROM HTML
"""
def get_data(root_folder):
    texts = []
    for i in os.listdir(root_folder):
        with open(os.path.join(root_folder, i), encoding="utf8") as f:
            soup = BeautifulSoup(f.read())
            content = ""
            for para in soup.find_all('p'):
                content += " "+para.text
            texts.append(cleanText(content))
    return texts



"""
TRAIN LDA (LATENT DIRICHLET ALLOCATION)
"""
def train_lda(data):
    num_topics = 300
    chunksize = 300
    dictionary = corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(doc) for doc in data]
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=2)
    return dictionary,corpus,lda


"""
Get Training data
"""
def get_training_data(tokens, dictionary, corpus, lda):
    X, y = [], []
    for ind in range(len(tokens)):
        if ind < len(pos_texts):
            y.append(1.)
        else:
            y.append(0.)
        bow = dictionary.doc2bow(tokens[ind])
        doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=bow)])
        X.append(doc_distribution)
    return np.array(X), np.array(y)


"""
F1 score metrics
"""
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def main():
    pos_texts = get_data(os.path.join("training", "positive"))
    neg_texts = get_data(os.path.join("training", "negative"))

    if EXPERIMENT:
        all_texts = []
        for i in pos_texts+neg_texts:
            sents = sent_tokenize(i.lower())
            texts = []
            for sent in sents:
                texts.append(word_tokenize(sent))
            all_texts.append(texts)
        all_sents = []
        for texts in all_texts:
            for sents in texts:
                all_sents.append(sents)
        w2v = Word2Vec(all_sents, size=100, window=5, min_count=1, workers=4)
        w2v.save("w2v.model")

    #TOKENIZE AND CLEAN DATA
    tokens = []
    for text in pos_texts+neg_texts:
        tokens.append(apply_all(text))

    words = [word for sent in tokens for word in sent]

    fdist = FreqDist(words)

    k = 100000
    top_k_words = fdist.most_common(k)
    top_k_words[-10:]

    top_k_words,_ = zip(*fdist.most_common(k))
    top_k_words = set(top_k_words)
    def keep_top_k_words(text):
        return [word for word in text if word in top_k_words]


    tokens = [keep_top_k_words(text) for text in tokens] #KEEPING TOP K Words


    #TRAINING LDA
    dictionary, corpus, lda = train_lda(tokens)

    #Generate Training Data
    X, y = get_training_data(tokens, dictionary, corpus, lda)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)


    #Make Sequential Fully Connected Neural Network Model
    dl_model = Sequential()
    activation = "relu"
    dl_model.add(Dense(128, input_shape=(300,), activation=activation))
    dl_model.add(Dense(64, activation=activation))
    dl_model.add(Dense(64, activation=activation))
    dl_model.add(Dense(32, activation=activation))
    dl_model.add(Dense(1, activation="sigmoid"))

    dl_model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0001), metrics=[get_f1])


    #Train model
    dl_model.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=10)

    #Test Score :
    preds = dl_model.predict(X_test)
    f1_score(y_test, (preds > 0.9).astype(np.float32))

    #Saving all the models
    print("Saving models")
    dl_model.save("classifier.h5")
    lda.save("lda.model")
    dictionary.save("corpora.dictionary")


if __name__ == '__main__':
    main()









