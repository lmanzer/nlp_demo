import os
from collections import Counter
import string 

import gensim
import networkx as nx
import nltk
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy


def remove_stopwords(sentence):
    word_list = sentence.split()
    _non_stopwords = " ".join([word for word in word_list if word not in stop_words])
    return _non_stopwords


def lemmatize_words(sentence):
    # we can probably do better than this! 
    _lemma_words = " ".join([word.lemma_ for word in nlp(sentence)])
    return _lemma_words
    

def clean_sentence_words(article_text):
    
    article_sentence_list = nltk.tokenize.sent_tokenize(cleaned_article)
    removed_punctuation = [w.translate(punctuation_table) for w in article_sentence_list]
    removed_mdash = [sentence.replace('—', '') for sentence in removed_punctuation]
    lower_sentences = [s.lower() for s in removed_mdash]
    removed_stopwords = [remove_stopwords(sentence) for sentence in lower_sentences]
    lemmatize_sentences = [lemmatize_words(sentence) for sentence in removed_stopwords]

    return lemmatize_sentences


def calculate_sentence_embedding

def document_summarization(article_text):
    cleaned_sentences = clean_sentence_words(article_text)


    return lemmatize_sentences

def nlp_analysis(article)
    article_text = article['article']

    # Clean Article Text
    cleaned_article = article_text.replace('\n', ' ')

    # Named-Entity Extraction
    nlp_data = nlp(cleaned_article)
    entities =  nlp_data.ents

    organizations = [ent.text.replace('\n','') for ent in entities if ent.label_ == 'ORG']
    people = [ent.text.replace('\n','') for ent in entities if ent.label_ == 'PERSON']

    top_orgs = Counter(organizations).most_common(10)
    top_people = Counter(people).most_common(10)


    return {
        'people': top_people,
        'organizations': top_orgs,
    }