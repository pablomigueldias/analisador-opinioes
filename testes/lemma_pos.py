import spacy

nlp = spacy.load('pt_core_news_sm')

texto = 'A vida é feita de escolhas. Eu escolhi a perseverança.'

doc = nlp(texto)

print(f'{'TOKEN':15} | {'LEMA':15} | {'POS':8}')
print('-'*40)

for t in doc:
    print(f'{t.text:15} | {t.lemma_:15} | {t.pos_:8}')
