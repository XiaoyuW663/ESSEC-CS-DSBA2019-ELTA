# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:52:00 2020

@author: wong
"""

import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm

#use this module to clean and convert the title texts into numerical vectors
class sentence2vector:
        def __init__(self,sentences,method = 'TF-IDF',vector_size = 100, min_count = 2):
            import spacy
            try:
                self.spacy_nlp = spacy.load("fr")
            except:
                #! spacy download fr
                self.spacy_nlp = spacy.load("fr_core_news_sm")

            self.unknown_token = '<ukn>'
            self.sentences = sentences
            self.method = method
            self.vector_size = vector_size
            self.min_count = min_count
            

            print('Size of documents:', len(self.sentences))
            print('Method of vectorization:', self.method)
            self.preprocessing()
            self.count_word()
            self.vectorize()
            


        def preprocessing(self):
            print('Preprocessing sentences...')
            try:
                with tqdm(self.sentences) as t:
                    for i,_ in enumerate(t):
                        self.sentences[i] = self.raw_to_tokens(self.sentences[i])
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()


        def normalize_accent(self,string):
                string = string.replace('á', 'a')
                string = string.replace('â', 'a')

                string = string.replace('é', 'e')
                string = string.replace('è', 'e')
                string = string.replace('ê', 'e')
                string = string.replace('ë', 'e')

                string = string.replace('î', 'i')
                string = string.replace('ï', 'i')

                string = string.replace('ö', 'o')
                string = string.replace('ô', 'o')
                string = string.replace('ò', 'o')
                string = string.replace('ó', 'o')

                string = string.replace('ù', 'u')
                string = string.replace('û', 'u')
                string = string.replace('ü', 'u')

                string = string.replace('ç', 'c')
        
                return string

        def raw_to_tokens(self,raw_string):
                # Write code for lower-casing
                string = raw_string.lower()
        
                # Write code to normalize the accents
                string = self.normalize_accent(string)
                
                # Write code to tokenize
                string = self.spacy_nlp(string)
                
                # Write code to remove punctuation tokens, stop words , digits and create string tokens
                string = [token.orth_ for token in string if not token.is_punct if not token.is_stop if token.orth_.isalpha()]
        
                # Write code to join the tokens back into a single string
                #clean_string = " ".join(string_tokens)
        
                return string


        def vectorize(self):
            if self.method    == 'TF-IDF':
                self.tfidf()
            if self.method == 'doc2vec':                
                self.doc2vec() 
            

        def tfidf(self):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import PCA
            print('Transform TF-IDF vectors...')
            #create a TfidfVectorizer object
            self.vectorizer = TfidfVectorizer(min_df=self.min_count)

            # vectorize the text
            x = [" ".join(sentence) for sentence in self.sentences]
            sparse_result = self.vectorizer.fit_transform(x)
            self.vocabulary = self.vectorizer.vocabulary_
            print('Vocabulary size:', len(self.vocabulary))     
            self.X = sparse_result.toarray()
            
            #reduce feature dimension of X
            pca = PCA(n_components=self.vector_size,copy = False)
            self.X = pca.fit_transform(self.X)


        def doc2vec(self):
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument

            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.sentences)]
            print('Training Doc2vec model...')
            self.vectorizer = Doc2Vec(vector_size=self.vector_size, window=5, min_count=self.min_count,hs=0,negative=5,
                                                            workers=-1, alpha=0.025, min_alpha=1e-5)
            self.vectorizer.build_vocab(documents)
            self.vocabulary = self.vectorizer.wv.vocab
            print('Vocabulary size:', len(self.vocabulary))
            self.vectorizer.train(documents,total_examples = self.vectorizer.corpus_count,epochs = self.vectorizer.epochs)
            self.X = np.array([self.vectorizer[i] for i in range(len(self.sentences))])

        def count_word(self):
            print('Building word2count dict...')
            self.word2count = {}
            try:
                with tqdm(self.sentences) as t:
                    for sentence in t:
                        for word in sentence:
                            if word in self.word2count:
                                self.word2count[word] += 1
                            else:
                                self.word2count[word] = 1
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

        def __getitem__(self,key):
            
            if self.method == 'TF-IDF':
                vec = self.vectorizer[key].toarray().squeeze()
            else:
                vec = self.vectorizer[key]
            return vec

        def __len__(self):
            return len(self.sentences)
