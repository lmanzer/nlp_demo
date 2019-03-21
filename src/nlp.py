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

# SETTINGS/CONSTANTS
# -------------------

# Load English tokenizer, tagger, parser, NER and word vectors
# English multi-task CNN trained on OntoNotes, with GloVe vectors trained on Common Crawl. 
# Assigns word vectors, context-specific token vectors, POS tags, dependency parse and named entities.
nlp = spacy.load('en_core_web_lg') 

# Stopwords
stop_words =  nltk.corpus.stopwords.words('english')
stop_words.remove('more')
stop_words.remove('against')

# Punctuation
punctuation_table = str.maketrans('', '', string.punctuation)


# FUNCTIONS
# -------------------

def remove_stopwords(sentence):
    word_list = sentence.split()
    _non_stopwords = " ".join([word for word in word_list if word not in stop_words])
    return _non_stopwords


def lemmatize_words(sentence):
    _lemma_words = " ".join([word.lemma_ for word in nlp(sentence)])
    return _lemma_words
    

def clean_sentence_words(article_sentences):
    
    removed_punctuation = [w.translate(punctuation_table) for w in article_sentences]
    removed_mdash = [sentence.replace('—', '') for sentence in removed_punctuation]
    lower_sentences = [s.lower() for s in removed_mdash]
    removed_stopwords = [remove_stopwords(sentence) for sentence in lower_sentences]
    lemmatize_sentences = [lemmatize_words(sentence) for sentence in removed_stopwords]

    return lemmatize_sentences


def calculate_similarity_matrix(article_sentences):
    similarity_matrix = np.zeros([len(article_sentences), 
                                len(article_sentences)])

    similarity_matrix
    for i, sentence_i in enumerate(article_sentences):
        nlp_i = nlp(sentence_i)
        for j, sentence_j in enumerate(article_sentences):
            if i != j:
                nlp_j = nlp(sentence_j)
                similarity_matrix[i][j] = nlp_i.similarity(nlp_j)

    return similarity_matrix


def generate_summary(article_sentences, similarity_matrix, N_SENTENCES=3):
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(
        ((scores[i], s, article_sentences[i]) for i,s in enumerate(article_sentences)), 
        reverse=True)
    
    generated_summary = [ranked_sentence[2] for ranked_sentence in ranked_sentences[0:N_SENTENCES]]

    return generated_summary


def document_summarization(cleaned_sentences, article_sentences):

    similarity_matrix = calculate_similarity_matrix(cleaned_sentences)
    generated_summary = generate_summary(article_sentences, similarity_matrix, N_SENTENCES=3)

    return generated_summary


def extract_entity(cleaned_article, entity_type):
 
    nlp_data = nlp(cleaned_article)
    entities =  nlp_data.ents

    entity = [ent.text.replace('\n','') for ent in entities if ent.label_ == entity_type]

    top_entity = Counter(entity).most_common(10)

    return top_entity


def topic_modeling(cleaned_sentences, N_TOPICS=3, N_WORDS=5):
    
    tokenized_sentences = [nltk.tokenize.word_tokenize(sentence) for sentence in cleaned_sentences]
    article_dictionary = gensim.corpora.Dictionary(tokenized_sentences)  # This needs describing!
    article_corpus = [article_dictionary.doc2bow(text) for text in tokenized_sentences] # This needs describing!

    article_ldamodel = gensim.models.ldamodel.LdaModel(article_corpus, 
                                                    num_topics = N_TOPICS, 
                                                    id2word=article_dictionary, 
                                                    passes=15)

    topics = article_ldamodel.print_topics(num_words=N_WORDS)

    return topics


def nlp_analysis(article):
    article_text = article['article']

    # Clean Article Text
    cleaned_article_pre = article_text.replace('\n', ' ')
    cleaned_article = cleaned_article_pre.replace(u'\xa0', u' ')

    article_sentences = nltk.tokenize.sent_tokenize(cleaned_article)

    # Entity Extraction
    top_orgs = extract_entity(cleaned_article, 'ORG')
    top_people = extract_entity(cleaned_article, 'PERSON')

    # Clean Sentences
    cleaned_sentences = clean_sentence_words(article_sentences)

    # Document Summarization
    document_summary = document_summarization(cleaned_sentences, article_sentences)

    # Topic Modelling
    topics = topic_modeling(cleaned_sentences)

    return {
        'organizations': top_orgs,
        'people': top_people,
        'pagerank_summary': document_summary,
        'topics': topics,
    }