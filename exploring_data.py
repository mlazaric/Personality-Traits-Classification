#!/usr/bin/env python

import spacy
import csv 
import os
import pickle

nlp = spacy.load('en_core_web_sm')
print("Loaded english spacy.")

processed_words_file = './.lemma_traits_pickles'

lemma_traits = []

if os.path.isfile(processed_words_file):
  with open(processed_words_file, 'rb') as f:
    lemma_traits = pickle.load(f) 
    
else:
  rows = []

  with open('./essays.csv', newline='', encoding='latin-1') as f:
    rows = list(csv.reader(f, delimiter=','))

  for idx, row in enumerate(rows[1:]):
    print(f'{idx + 1}/{len(rows) - 1}')
    traits = [t == 'y' for t in row[2:]]
    lemma_traits += [(token.lemma_, traits) for token in nlp(row[1])]

  with open(processed_words_file, 'wb') as f:
    pickle.dump(lemma_traits, f)

print(f'Number of non distinct tokens: {len(lemma_traits)}.')

from collections import defaultdict

ratios = [(defaultdict(int), defaultdict(int)) for _ in range(len(lemma_traits[0][1]))]

for lemma, traits in lemma_traits:
  for idx, trait in enumerate(traits):
    ratios[idx][trait][lemma] += 1

from math import log, e

significant_lemmas = [{} for _ in range(len(lemma_traits[0][1]))]

for idx, ny in enumerate(ratios):
  n, y = ny
  for lemma, count in y.items():
    significant_lemmas[idx][lemma] = (max(count - n[lemma], 0.0)) / log(n[lemma] + e)

for i, curr_lemmas in enumerate(significant_lemmas):
  for lemma, score in curr_lemmas.items():
    for j, other_lemmas in enumerate(significant_lemmas):
      if i != j and lemma in other_lemmas:
        other_lemmas[lemma] /= (1 + score) # -=
  
scores = [sorted(list(m.items()), reverse=True, key=lambda t: t[1]) 
  for m in significant_lemmas]

for score in scores:
  print([word for word, _ in score[:10]])
