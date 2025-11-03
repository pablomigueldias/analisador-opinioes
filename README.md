# Analisador de Opiniões (PT-BR)

Um analisador de sentimentos e aspectos em textos de opinião em português, baseado em **spaCy** e **Transformers**.  
Ele identifica o **sentimento geral (positivo, negativo ou neutro)**, detecta **aspectos** (como “bateria”, “preço”, “tela”) e analisa **sentenças individualmente**.

---

## Funcionalidades

-  Análise de sentimento em português usando modelo **BERTweet PT-BR**  
- Detecção de **aspectos** via dependência sintática e regras por vocabulário  
-  Reconhecimento de entidades nomeadas (NER)  
-  Suporte a entrada `.txt` ou `.csv`  
-  Gera saída estruturada em `.json` com todos os resultados  
-  Pipeline totalmente offline após primeiro download do modelo

---

## 🧰 Tecnologias

- [spaCy](https://spacy.io/)
- [Transformers (Hugging Face)](https://huggingface.co/transformers/)
- [pysentimiento/bertweet-pt-sentiment](https://huggingface.co/pysentimiento/bertweet-pt-sentiment)

---

## Instalação

Crie e ative o ambiente virtual:

```bash
python -m venv venv
venv\Scripts\activate
```
```bash
pip install spacy transformers pysentimiento
python -m spacy download pt_core_news_sm
```
Rodar o script
```bash
python analisador_cli.py --in "avaliacoes.txt" --out "resultado.json"
```
## Dica
Se estiver sem espaço no C: (para os modelos da Hugging Face), você pode definir um cache personalizado dentro do projeto
```Python
os.environ["HF_HOME"] = r"F:\hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"F:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"F:\hf_cache"
```

