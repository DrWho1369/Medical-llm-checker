import streamlit as st
import openai
import requests
import pandas as pd
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import re
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")  

def get_llm_response(query):
    """Fetch response from an LLM (GPT-4)."""
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

def fetch_pubmed_articles(query):
    """Fetch articles from PubMed."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": 5}
    response = requests.get(base_url, params=params)
    article_ids = response.json().get("esearchresult", {}).get("idlist", [])
    
    articles = []
    for article_id in article_ids:
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": article_id, "rettype": "abstract", "retmode": "text"}
        article_response = requests.get(fetch_url, params=fetch_params)
        articles.append(article_response.text)
    
    return articles

def fetch_google_scholar_articles(query):
    """Fetch articles from Google Scholar (via Semantic Scholar API)."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,abstract"
    response = requests.get(url)
    data = response.json().get("data", [])
    return [entry.get("abstract", "No abstract available") for entry in data]

def get_medical_sources(query):
    """Fetch data from multiple sources."""
    sources = {
        "WHO": f"https://www.who.int/api/some-endpoint?q={query}",
        "CDC": f"https://data.cdc.gov/resource/some-dataset.json?query={query}",
        "Wikipedia": f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}",
        "Europe PMC": f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&format=json",
        "Google Scholar": f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,abstract"
    }
    results = {}
    for name, url in sources.items():
        response = requests.get(url)
        results[name] = response.text if response.status_code == 200 else "No data found"
    
    # Fetch PubMed articles separately
    pubmed_articles = fetch_pubmed_articles(query)
    results["PubMed"] = " ".join([text for text in pubmed_articles]) if pubmed_articles else "No data found"
    
    return results

def detect_bias(text):
    """Detect potential bias using NLTK for tokenization."""
    bias_indicators = {
        "Gender": ["male", "female", "men", "women", "transgender", "non-binary"],
        "Race": ["Black", "White", "Asian", "Hispanic", "Caucasian", "Indigenous"],
        "Age": ["elderly", "young", "children", "teenager", "middle-aged"],
        "Socioeconomic": ["rich", "poor", "low-income", "high-income", "privileged", "underprivileged"]
    }
    
    detected_bias = {}
    highlighted_text = text
    sentences = sent_tokenize(text)  # Tokenize sentences
    
    for category, terms in bias_indicators.items():
        for sentence in sentences:
            words = word_tokenize(sentence)  # Tokenize words
            for term in terms:
                if term in words:
                    detected_bias[category] = detected_bias.get(category, []) + [(term, sentence)]
                    highlighted_text = highlighted_text.replace(term, f"**{term}**")
    
    return detected_bias, highlighted_text


def plot_bias_distribution(bias_data, title):
    """Generate a bar chart showing bias distribution."""
    if bias_data:
        categories = list(bias_data.keys())
        counts = [len(bias_data[cat]) for cat in categories]
        
        plt.figure(figsize=(6, 4))
        plt.bar(categories, counts, color=["red", "blue", "green", "purple"])
        plt.xlabel("Bias Categories")
        plt.ylabel("Occurrences")
        plt.title(title)
        st.pyplot(plt)
    else:
        st.info("No bias detected.")

# Streamlit App UI
st.set_page_config(page_title="Medical LLM Bias & Misinformation Checker", layout="wide")
st.title("🔍 Medical LLM Bias & Misinformation Checker")

query = st.text_input("🩺 **Enter a medical question:**", help="Analyze AI accuracy and bias.")

if st.button("🔎 Analyze Response"):
    if query:
        with st.spinner("Fetching response and analyzing..."):
            llm_response = get_llm_response(query)
            medical_sources = get_medical_sources(query)
            
            bias_detected_llm, highlighted_llm = detect_bias(llm_response)
            bias_detected_sources = {source: detect_bias(text)[0] for source, text in medical_sources.items()}
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("🤖 LLM Response:")
                st.markdown(highlighted_llm, unsafe_allow_html=True)
                plot_bias_distribution(bias_detected_llm, "Bias in LLM Response")
            
            with col2:
                st.subheader("📜 Medical Sources:")
                for source_name, content in medical_sources.items():
                    st.markdown(f"### {source_name}")
                    st.markdown(content[:500] + "...") if content != "No data found" else st.warning(f"No data found for {source_name}.")
                    plot_bias_distribution(bias_detected_sources[source_name], f"Bias in {source_name}")
    else:
        st.warning("Please enter a medical question.")
