import json
from typing import List,Dict,Any
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('pt_core_news_sm')

texto = 'A vida é feita de escolhas. Eu escolhi a perseverança.'

doc = nlp(texto)

for t in doc:
    print(t.text, end=' | ')
print()
