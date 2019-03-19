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

# GloVe is an unsupervised learning algorithm for obtaining vector representations for words. 
# Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and 
# the resulting representations showcase interesting linear substructures of the word vector space.
# https://nlp.stanford.edu/projects/glove/
word_embeddings = {}
f = open(os.path.join('data', 'models', 'external', 'glove', 'glove.6B.100d.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

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


def calculate_sentence_embedding(sentences):
    empty_vector = np.zeros((100,))

    sentence_vectors = []
    for sentence in sentences:
        if len(sentence) > 0:
            _word_list = sentence.split()
            _word_vector =  [word_embeddings.get(word, empty_vector) for word in _word_list]
            _summed_vector = sum(_word_vector)
            normalized_vector = _summed_vector/ (len(sentence.split()))
        else:
            normalized_vector = empty_vector
        sentence_vectors.append(normalized_vector)

    return sentence_vectors


def calculate_similarity_matrix(sentence_vectors, article_sentences):
    sim_mat = np.zeros([len(article_sentences), len(article_sentences)])

    for i in range(len(article_sentences)):
        for j in range(len(article_sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), 
                                                sentence_vectors[j].reshape(1,100))[0,0]

    return sim_mat


def generate_summary(article_sentences, similarity_matrix, N_SENTENCES=3):
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(
        ((scores[i], s, article_sentences[i]) for i,s in enumerate(article_sentences)), 
        reverse=True)
    
    generated_summary = [ranked_sentence[2] for ranked_sentence in ranked_sentences[0:N_SENTENCES]]

    return generated_summary


def document_summarization(cleaned_sentences, article_sentences):

    sentence_vectors = calculate_sentence_embedding(cleaned_sentences)
    similarity_matrix = calculate_similarity_matrix(sentence_vectors, article_sentences)
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
    cleaned_article = article_text.replace('\n', ' ')
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