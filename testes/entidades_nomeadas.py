import spacy

nlp = spacy.load('pt_core_news_sm')

texto = 'A vida é feita de escolhas. Eu escolhi a perseverança.'

doc = nlp(texto)

print(f"{'ENTIDADE':30} | {'TIPO':10}")
print("-" * 45)
for ent in doc.ents:
    print(f"{ent.text:30} | {ent.label_:10}")

print("\nExplicações dos tipos encontrados:")
rotulos = sorted(set(e.label_ for e in doc.ents))
for r in rotulos:
    print(f"{r:10} -> {spacy.explain(r)}") #type: ignore
