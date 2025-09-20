# server_local.py — uses a local transformers NLI model
import os, math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# choose a lighter NLI model if you are on CPU: typeform/distilbert-base-uncased-mnli
MODEL_NAME = "typeform/distilbert-base-uncased-mnli"  # smaller than roberta-large-mnli
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

LABELS = ["contradiction", "neutral", "entailment"]  # typical ordering for many MNLI models

def nli_snippet_vs_claim(snippet, claim):
    # treat snippet as premise, claim as hypothesis: entailment => snippet supports claim
    inputs = tokenizer(snippet, claim, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
    # softmax
    import numpy as np
    probs = np.exp(logits) / np.sum(np.exp(logits))
    # map to labels
    idx = int(probs.argmax())
    label = LABELS[idx]
    score = float(probs[idx])
    # normalize to named labels: entailment -> support
    if label == "entailment":
        return {"label":"supports","score":score}
    elif label == "contradiction":
        return {"label":"contradicts","score":score}
    else:
        return {"label":"neutral","score":score}

# reuse helper from Option A for extracting snippet and domain trust
DOMAIN_TRUST = {"reuters.com":0.95,"apnews.com":0.92,"bbc.com":0.9}
def domain_trust(domain):
    dom = (domain or "").lower().replace("www.","")
    for k,v in DOMAIN_TRUST.items():
        if k in dom: return v
    return 0.5

def extract_snippet_text(url):
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text,"html.parser")
        article = soup.find("article")
        if article:
            txt = " ".join(p.get_text(" ", strip=True) for p in article.find_all("p"))
        else:
            ps = soup.find_all("p")
            txt = " ".join(p.get_text(" ", strip=True) for p in ps[:6])
        return (txt or "")[:800]
    except:
        return ""

def search_duckduckgo(query, max_results=8):
    """Search using the updated DuckDuckGo DDGS API"""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
    return results

class Req(BaseModel):
    claim: str

@app.post("/verify")
def verify(req: Req):
    claim = (req.claim or "").strip()
    if not claim:
        raise HTTPException(status_code=400, detail="claim required")

    results = search_duckduckgo(claim, max_results=8)
    if not results:
        return {"score": None, "verdict": "No evidence", "explanation": "No search results", "sources": []}

    sources=[]
    support_w = contradict_w = 0.0
    for r in results[:8]:
        title = r.get("title") or r.get("text","")
        url = r.get("href") or r.get("url","")
        snippet = r.get("body") or r.get("snippet") or ""
        if len(snippet) < 80 and url:
            snippet = extract_snippet_text(url)[:600] or snippet

        out = nli_snippet_vs_claim(snippet or title, claim)
        lbl = out["label"]; sc = out["score"]
        if lbl == "supports":
            sign = 1
        elif lbl == "contradicts":
            sign = -1
        else:
            sign = 0
        trust = domain_trust(url)
        weight = trust * (0.6 + 0.4*sc)
        if sign > 0: support_w += weight
        elif sign < 0: contradict_w += weight

        sources.append({"title":title,"url":url,"snippet":snippet[:400],"stance":lbl,"score":sc,"weight":round(weight,3)})

    total = support_w + contradict_w
    if total == 0:
        return {"score": None, "verdict":"Insufficient evidence","explanation":"No clear support/contradict found","sources":sources}
    credibility = int(round((support_w / total) * 100))
    verdict = "Likely true" if credibility>=66 else ("Likely false" if credibility<=34 else "Ambiguous / mixed")
    return {"score":credibility,"verdict":verdict,"explanation":f"Aggregated {len(sources)} sources (local NLI).","sources":sources}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_local:app", host="0.0.0.0", port=8000, reload=True)
