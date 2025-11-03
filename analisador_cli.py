# -*- coding: utf-8 -*-

# ====== CONFIG DE CACHE (opcional) ======
import os
os.environ["HF_HOME"] = r"F:\hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"F:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"F:\hf_cache"

# ====== IMPORTS ======
import csv
import argparse
import json
from typing import List, Dict, Any

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from transformers import pipeline



def init_nlp() -> spacy.Language:  #type: ignore
    nlp = spacy.load("pt_core_news_sm")
    return nlp  


def init_sentiment():
    sent_clf = pipeline( #type: ignore
        "sentiment-analysis", #type: ignore
        model="pysentimiento/bertweet-pt-sentiment"
    )
    return sent_clf 


nlp = init_nlp()
sent_clf = init_sentiment()

def sentimento_pt(texto: str) -> Dict[str, float]:
    out = sent_clf(texto)[0] #type: ignore
    return {"label": out["label"], "score": float(out["score"])}


def aspectos_por_dependencia(doc: Doc) -> List[Dict[str, Any]]:
    pares = []
    for token in doc:
        if token.pos_ == "VERB":
            objetos = [w for w in token.children if w.dep_ in ("obj", "dobj")]
            if objetos:
                pares.append({
                    "verbo": token.lemma_,
                    "objetos": [o.text for o in objetos]
                })
    return pares


def aspectos_copulares(doc: Doc) -> List[Dict[str, str]]:
    pares = []
    for t in doc:
        if t.pos_ == "ADJ":
            sujeitos = [w for w in t.children if w.dep_ in ("nsubj", "nsubj:pass") and w.pos_ in ("NOUN", "PROPN")]
            copulas = [w for w in t.children if w.dep_ == "cop" and w.lemma_ in ("ser", "estar")]
            if sujeitos and copulas:
                pares.append({
                    "aspecto": sujeitos[0].text,
                    "qualidade": t.text,
                    "verbo_copula": copulas[0].text
                })
    return pares


def construir_matcher(nlp_obj: spacy.Language) -> Matcher: #type: ignore
    termos = ["bateria", "câmera", "camera", "tela", "som", "teclado",
              "desempenho", "acabamento", "preço", "memória", "processador"]
    matcher = Matcher(nlp_obj.vocab)
    padroes = []
    for termo in termos:
        padroes.append([{"LOWER": termo}])                  
        padroes.append([{"LOWER": termo}, {"POS": "ADJ"}])  
    return matcher


matcher = construir_matcher(nlp)


def analisar_opiniao(texto: str) -> Dict[str, Any]:
    
    doc = nlp(texto)
 
    sent_global = sentimento_pt(texto)

    entidades = [{"texto": ent.text, "tipo": ent.label_} for ent in doc.ents]

    aspectos_dep = aspectos_por_dependencia(doc)
    aspectos_cop = aspectos_copulares(doc)

    spans = [doc[s:e] for _, s, e in matcher(doc)]
    aspectos_regrados = [span.text for span in spacy.util.filter_spans(spans)] 

    sentencas = []
    for sent in doc.sents:
        out = sentimento_pt(sent.text)
        sentencas.append({
            "texto": sent.text,
            "label": out["label"],
            "score": round(out["score"], 3)
        })

    return {
        "texto": texto,
        "sentimento": {
            "label": sent_global["label"],
            "score": round(sent_global["score"], 3)
        },
        "entidades": entidades,
        "aspectos": {
            "por_dependencia": aspectos_dep,
            "por_copula": aspectos_cop,
            "por_regras": aspectos_regrados
        },
        "sentencas": sentencas
    }


def ler_txt(caminho: str) -> List[str]:
    with open(caminho, "r", encoding="utf-8-sig", errors="ignore") as f:
        return [linha.strip() for linha in f if linha.strip()]


def ler_csv(caminho: str, coluna: str) -> List[str]:
    textos: List[str] = []
    with open(caminho, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if coluna not in reader.fieldnames: #type: ignore
            raise ValueError(f'Coluna "{coluna}" nao encontrada. Colunas: {reader.fieldnames}')
        for row in reader:
            t = (row.get(coluna) or "").strip()
            if t:
                textos.append(t)
    return textos

def salvar_json(dados: Any, caminho: str):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Analisador de Opiniões (PT-BR) - spaCy + Transformers")
    p.add_argument("--in", dest="entrada", required=True, help="Arquivo de entrada (.txt ou .csv)")
    p.add_argument("--out", dest="saida", required=True, help="Arquivo .json de saída")
    p.add_argument("--coluna", dest="coluna", help="(CSV) nome da coluna de texto")
    return p.parse_args()

def main():
    args = parse_args()
    caminho_in = args.entrada
    caminho_out = args.saida

    ext = os.path.splitext(caminho_in)[1].lower()
    if ext == ".txt":
        textos = ler_txt(caminho_in)
    elif ext == ".csv":
        if not args.coluna:
            raise SystemExit("Para CSV, informe --coluna NOME_DA_COLUNA")
        textos = ler_csv(caminho_in, args.coluna)
    else:
        raise SystemExit("Formato não suportado. Use .txt ou .csv")

    resultados = [analisar_opiniao(t) for t in textos]
    salvar_json(resultados, caminho_out)
    print(f"Processado {len(resultados)} textos. Salvo em: {caminho_out}")


if __name__ == "__main__":
    main()
