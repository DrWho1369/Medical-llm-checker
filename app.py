import streamlit as st
import openai
import requests
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import re

def get_llm_response(query):
    """Fetch response from an LLM (GPT-4)."""
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    )
    return response["choices"][0]["message"]["content"]

def get_pubmed_articles(query):
    """Fetch related medical articles from PubMed."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 5  # Get top 5 related articles
    }
    response = requests.get(base_url, params=params)
    article_ids = response.json().get("esearchresult", {}).get("idlist", [])
    
    articles = []
    for article_id in article_ids:
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": article_id, "rettype": "abstract", "retmode": "text"}
        article_response = requests.get(fetch_url, params=fetch_params)
        articles.append((article_id, article_response.text))
    
    return articles

def check_similarity(llm_response, medical_articles):
    """Check similarity between LLM response and trusted medical sources."""
    model = SentenceTransformer("allenai/specter")  # Pretrained biomedical sentence embedding model
    response_embedding = model.encode(llm_response, convert_to_tensor=True)
    scores = []
    citations = []
    
    for article_id, article_text in medical_articles:
        article_embedding = model.encode(article_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(response_embedding, article_embedding).item()
        scores.append((article_id, similarity))
    
    scores.sort(key=lambda x: x[1], reverse=True)  # Sort by highest similarity
    top_citations = scores[:3]  # Get top 3 most relevant sources
    avg_similarity = sum([s[1] for s in top_citations]) / len(top_citations) if top_citations else 0
    
    for article_id, similarity in top_citations:
        citations.append(f"PubMed ID: [{article_id}](https://pubmed.ncbi.nlm.nih.gov/{article_id}) - Similarity: {similarity:.2f}")
    
    return avg_similarity, citations

def detect_bias(llm_response):
    """Detect potential demographic bias in the LLM response and highlight flagged terms."""
    bias_indicators = {
        "gender": ["male", "female", "men", "women", "transgender", "non-binary"],
        "race": ["Black", "White", "Asian", "Hispanic", "Caucasian", "Indigenous"],
        "age": ["elderly", "young", "children", "teenager", "middle-aged"],
        "socioeconomic": ["rich", "poor", "low-income", "high-income", "privileged", "underprivileged"]
    }
    
    detected_bias = {}
    highlighted_response = llm_response
    for category, terms in bias_indicators.items():
        for term in terms:
            if re.search(rf"\b{term}\b", llm_response, re.IGNORECASE):
                detected_bias[category] = detected_bias.get(category, 0) + 1
                highlighted_response = re.sub(rf"\b{term}\b", f"**{term}**", highlighted_response, flags=re.IGNORECASE)
    
    return detected_bias, highlighted_response

def save_feedback(query, response, feedback):
    """Save user feedback to a CSV file."""
    feedback_data = pd.DataFrame([[query, response, feedback]], columns=["Query", "Response", "Feedback"])
    try:
        feedback_data.to_csv("feedback.csv", mode='a', header=False, index=False)
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

# Streamlit App UI
st.title("Medical LLM Bias & Misinformation Checker")
query = st.text_input("Enter a medical question:")
feedback = st.text_area("Provide feedback on the response (optional):")
feedback_submitted = st.button("Submit Feedback")

if feedback_submitted:
    if feedback:
        save_feedback(query, "LLM response here", feedback)  # Placeholder for actual response
        st.success("Thank you for your feedback! Your input helps improve accuracy.")
    else:
        st.warning("Please enter feedback before submitting.")

if st.button("Check Response"):
    if query:
        with st.spinner("Fetching response..."):
            llm_response = get_llm_response(query)
            medical_articles = get_pubmed_articles(query)
            similarity_score, citations = check_similarity(llm_response, medical_articles)
            bias_detected, highlighted_response = detect_bias(llm_response)
            
            st.subheader("LLM Response:")
            st.markdown(highlighted_response, unsafe_allow_html=True)
            
            st.subheader("Trustworthiness Score:")
            st.write(f"Similarity to trusted medical sources: {similarity_score:.2f}")
            
            if similarity_score < 0.5:
                st.warning("⚠️ This response may contain misleading or biased information.")
            else:
                st.success("✅ This response closely matches trusted medical sources.")
            
            st.subheader("Citations from Trusted Sources:")
            if citations:
                for citation in citations:
                    st.markdown(citation)
            else:
                st.write("No relevant citations found.")
            
            st.subheader("Bias Detection:")
            if bias_detected:
                bias_report = "Potential bias detected in the following categories: " + ", ".join(bias_detected.keys())
                st.warning(bias_report)
            else:
                st.success("No explicit bias detected in this response.")
    else:
        st.warning("Please enter a medical question.")
