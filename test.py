import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from sklearn.tree import DecisionTreeClassifier
from operator import itemgetter
from itertools import product

dataset = pd.read_csv('essays.csv')
porter_stemmer = PorterStemmer()

# print(dataset.groupby(['cNEU', 'cAGR']).size())
#features = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']

#for a, b in product(features, features):
#    if a == b: continue
#
#    print(pd.crosstab(dataset[a], dataset[b], normalize = 'columns'))
#    print()

dataset['text_without_stop_words'] = [remove_stopwords(text) for text in dataset['TEXT']]
dataset['preprocessed_text_tokens'] = [simple_preprocess(text, deacc = True) for text in dataset['text_without_stop_words']]
dataset['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in dataset['preprocessed_text_tokens']]

dictionary = Dictionary(dataset['stemmed_tokens'])
#dictionary.filter_n_most_frequent(10)

bows = [corpus2csc([dictionary.doc2bow(tokens)], num_terms = len(dictionary)).toarray()[:, 0] for tokens in dataset['preprocessed_text_tokens']]
# bows = [[word_count / len(tokens) for word_count in bow] for tokens, bow in zip(dataset['preprocessed_text_tokens'], bows)]

bow_clf = DecisionTreeClassifier(random_state = 0)

bow_clf.fit(bows, dataset['cNEU'])

importances = list(bow_clf.feature_importances_)

feature_importances = [(feature, round(importance, 10)) for feature, importance in zip(dictionary.token2id.keys(), importances)]
feature_importances = sorted(feature_importances, key = itemgetter(1), reverse = True)

for _, pair in zip(range(20), feature_importances):
    print(f'Variable: {pair[0]:20} Importance: {pair[1]}')
