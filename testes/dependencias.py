import spacy
from spacy import displacy

nlp = spacy.load('pt_core_news_sm')

texto = 'A vida é feita de escolhas. Eu escolhi a perseverança.'

doc = nlp(texto)

print(f'{'TOKEN':15} | {'DEP':15} | {'HEAD':15} | {'POS':6}')
print('-'*40)
for t in doc:
    print(f'{t.text:15} | {t.dep_:12} | {t.head.text:15} | {t.pos_}')

pares = []
for token in doc:
    if token.pos_ == 'VERB':
        objetos = [w for w in token.children if w.dep_ in ('obj', 'dobj')]
        if objetos:
            pares.append((token.lemma_, [o.text for o in objetos]))

print(f'\nPares verbo -> objeto:')
for verbo, objs in pares:
    print(f'  {verbo} -> {objs}')
