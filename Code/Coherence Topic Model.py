
# coding: utf-8

# In[ ]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
from nltk.corpus import stopwords
from collections import Counter
import string
from collections import Counter
import operator
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
import pickle


# In[ ]:


eudata = pd.read_csv('euData.csv') 
eudata['date'] = pd.to_datetime(eudata['date'])
eudata_early = eudata[eudata.date < '2001-01-01']
rownr = eudata_early.shape[0]
rownr


# In[ ]:


pd.options.mode.chained_assignment = None

wordCount = Counter([])
mepDict = {}

stopWords = list(nltk.corpus.stopwords.words('english'))

for i in range(rownr):
    if not i%10000:
        print(i)
    text = eudata_early['text'].iloc[i]
    agenda = eudata_early['agenda'].iloc[i]
    #remove punctuation, digits and lowering case
    text = (agenda + ' ' + text).translate(str.maketrans('','',(string.punctuation + string.digits + '–'))).lower()
     #removing unicode
    text = (text.encode('ascii', 'ignore')).decode('utf-8')
    #splitting text into word list
    textList = [x for x in text.split() if x not in stopWords]
    
    mep = eudata_early['name'].iloc[i]
    party = eudata_early['party'].iloc[i]
    euparty = eudata_early['euparty'].iloc[i]
    date = eudata_early['date'].iloc[i]
    speechnr = eudata_early['speechnr'].iloc[i]
    session = pd.Timestamp(date.year,date.month,1)
    k = (mep, session)
    if k in mepDict.keys():
        mepDict[k] = mepDict[k] + textList
    else:
        mepDict[k] = textList
    wordCount = wordCount + Counter(textList)


# In[ ]:


topWords = list(dict(sorted(wordCount.items(), key=operator.itemgetter(1), reverse=True)[:30]).keys())
topWords


# In[ ]:


len(mepDict)


# In[ ]:


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def stem(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]


# In[ ]:


fillWords = ['also', 'behalf', 'commission', 'commissioner', 'committee',
             'council', 'debate', 'european', 'gentlemen', 'item', 'ladies',
             'like', 'madam', 'make', 'minutes', 'mr', 'mrs', 'next', 
             'parliament', 'point', 'presidency', 'president', 'proposal',
             'question', 'say', 'sitting', 'thank', 'think', 'vote', 'want']
excludeWords = topWords + fillWords + stopWords
excludeWords = stem([get_lemma(x) for x in excludeWords])

mepDict_clean = {}

i = 0
for k in mepDict.keys():
    if not i % 1000:
        print(i)
    i += 1
    textList = mepDict[k]
    textList_clean = stem([get_lemma(x) for x in textList])
    textList_clean = [x for x in textList_clean if x not in excludeWords and len(x) > 2]
    if len(textList_clean) > 10:
        mepDict_clean[k] = textList_clean


# In[ ]:


len(mepDict_clean)


# In[ ]:


text_data = []

i = 0
for k in mepDict_clean.keys():
    if not i % 1000:
        print(i)
    i += 1
    text_data.append(mepDict_clean[k])


# In[ ]:


dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
pickle.dump(corpus, open('Trial_Topics/corpus.pkl', 'wb'))
dictionary.save('Trial_Topics/dictionary.gensim')


# # performance measurement

# In[ ]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[ ]:


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=text_data,limit=100,start=20, step=10)


# In[ ]:


limit=100; start=20; step=10;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[ ]:


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[ ]:


ldamodel = model_list[4]
ldamodel.save('Trial_Topics/model.gensim')
topics = ldamodel.print_topics(num_words=6)
for topic in topics:
    print(topic)


# In[ ]:


def get_topic(text):
    if type(text).__name__ == 'str':
        text = text.split()
    topics = ldamodel.get_document_topics(dictionary.doc2bow(text))
    return topics


# In[ ]:


topicData = np.array([(k[0], k[1], get_topic(v)) for k, v in mepDict_clean.items()]).reshape((-1,3))
topicData[0,:]


# In[ ]:


topicDF = pd.DataFrame(topicData, columns=['name', 'date', 'topic']).sort_values('date')
#topicDF['topic'] = pd.to_numeric(topicDF['topic'])
topicDF['date'] = pd.to_datetime(topicDF['date'])
topicDF['name'] = topicDF.name.apply(str)


# In[ ]:


topicDF.to_csv('Trial_Topics/topics.csv')

