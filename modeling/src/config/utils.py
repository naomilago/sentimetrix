from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
from loguru import logger
import matplotlib as mpl
import pandas as pd
import unicodedata
import numpy as np
import gensim
import string
import spacy
import emoji
import nltk
import copy
import os
import re

from typing import (
  Iterable, 
  List, 
  Mapping, 
  Tuple, 
  Union
)

from itertools import (
  chain, 
  dropwhile, 
  takewhile
)

mpl.rcParams['figure.dpi'] = 400

STOPWORDS = set(stopwords.words('english'))

DATA_INPUTS = dict({
  'SPACY_MODEL': 'en_core_web_trf'
})

DATA_PROCESS = dict({
  'POS_COLS': [
    'NOUN', 
    'PROPN', 
    'DET', 
    'ADJ', 
    'ADP', 
    'PRON', 
    'ADV', 
    'VERB', 
    'SCONJ', 
    'CCONJ', 
    'NUM', 
    'AUX', 
    'X', 
    'INTJ', 
    'PUNCT', 
    'SPACE', 
    'SYM', 
    'PART'
  ],
})

def demojize(msg: str) -> str:
    return emoji.demojize(msg, language = 'pt').replace(r":", " ")

def remove_punctuation(text: str) -> str:
    if type(text) == float:
        return text
    
    msg = ""
    
    for char in text:     
        if char not in string.punctuation:
            msg+=char
        else:
            msg+=" "
            
    return re.sub(r"\s+", " ", msg)
  
def remove_accents(msg: str) -> str:
    words = str(msg).split("\s")
    words = [unicodedata.normalize('NFKD', word) for word in words]
    words = [
        u"".join([c])
        for word in words
        for c in word
        if not unicodedata.combining(c)
    ]

    return "".join(words).lower()

def clean_msg(message: str) -> str:
    msg_copy = str(copy.copy(message))
    msg_copy = msg_copy.replace("\n", " ")
    msg_copy = msg_copy.replace("\n", "")
    msg_copy = remove_punctuation(msg_copy)
    msg_copy = demojize(msg_copy)
    msg_copy = remove_accents(msg_copy)
    
    return str(msg_copy)
  
def ngrams(text: str, ngram: int = 2, stopwords: List[str] = STOPWORDS) -> List[str]:
    if ngram == 1:
        words = [word for word in text.split(" ") if word not in stopwords]
    else:
        words = [word for word in text.split(" ")]
        
    temp = zip(*[words[i:] for i in range(0, ngram)])
    
    return ['_'.join(ngram) for ngram in temp]

def preprocessing(df: pd.DataFrame, message: str = 'text', nlp: spacy.lang.en.English = spacy.load(DATA_INPUTS['SPACY_MODEL']), stopwords: List[str] = STOPWORDS) -> pd.DataFrame:
        df["processed_messages"] = df.loc[:, message].apply(clean_msg)
        processed_messages = list()
        pos_msgs = list()
        counts_msgs = list()
        
        for doc in nlp.pipe(df.loc[:, "processed_messages"]):
            tokens = list()
            pos = list()
            count = list()
            
            for token in doc:
                if  not token.like_url \
                and not token.is_punct \
                and not token.is_space \
                and not token.like_num:
                    tokens.append(token.lemma_.lower())
                    pos.append(token.pos_)
                
            processed_messages.append(" ".join(tokens))
            pos_msgs.append(Counter(pos))
            
            try:
                counts_msgs.append([len(processed_messages[-1]), len(tokens[-1])])
            except IndexError:
                try:
                    counts_msgs.append([len(processed_messages[-1]), 0])
                except IndexError:
                    try:
                        counts_msgs.append([0, len(tokens[-1])])
                    except IndexError:
                        counts_msgs.append([0, 0])
                        
        POS_COLS = DATA_PROCESS['POS_COLS']
        df['processed_messages'] = processed_messages
        df.loc[:, 'processed_messages'] = df.loc[:, 'processed_messages'].apply(remove_accents)
        df['bigrams'] = df.loc[:, 'processed_messages'].apply(ngrams, args = (2, ))
        df['trigrams'] = df.loc[:, 'processed_messages'].apply(ngrams, args = (3, ))
        df.loc[:, 'processed_messages'] = df.loc[:, 'processed_messages'].apply(ngrams, args = (1, ))
        df.loc[:, 'processed_messages'] = df.apply(lambda x: list(chain(x['processed_messages'], x['bigrams'], x['trigrams'])), axis = 1)
        df.loc[:, 'processed_messages'] = df.loc[:, 'processed_messages'].apply(lambda x: " ".join(x))
        df_pos = pd.DataFrame(pos_msgs, index = df.index.tolist(), columns = POS_COLS)
        df_cnt = pd.DataFrame(counts_msgs, index = df.index.tolist(), columns = ['char_count', 'word_count'])
        
        return pd.concat([df.drop(columns = ['bigrams', 'trigrams']), df_pos, df_cnt], axis = 1).fillna(0) 