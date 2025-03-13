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
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

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
    pubmed_articles = get_pubmed_articles(query)
    results["PubMed"] = " ".join([text for _, text in pubmed_articles]) if pubmed_articles else "No data found"
    
    return results

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

def check_similarity(llm_response, medical_sources):
    """Check similarity between LLM response and multiple trusted medical sources."""
    model = SentenceTransformer("allenai/specter", device="cpu")
    response_embedding = model.encode(llm_response, convert_to_tensor=True)
    scores = []
    citations = []
    
    for source_name, source_text in medical_sources.items():
        source_embedding = model.encode(source_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(response_embedding, source_embedding).item()
        scores.append((source_name, similarity, source_text[:200]))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    top_citations = scores[:3]
    avg_similarity = sum([s[1] for s in top_citations]) / len(top_citations) if top_citations else 0
    
    for source_name, similarity, snippet in top_citations:
        citations.append(f"**{source_name}** - Similarity: {similarity:.2f}\n*Snippet:* {snippet}...")
    
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
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False

# Streamlit App UI
st.set_page_config(page_title="Medical LLM Bias & Misinformation Checker", layout="wide")
st.title("🔍 Medical LLM Bias & Misinformation Checker")
st.markdown("""<style>
    .stTextInput, .stButton, .stMarkdown, .stAlert {
        font-size: 18px !important;
    }
</style>""", unsafe_allow_html=True)

query = st.text_input("🩺 **Enter a medical question:**", help="Ask a medical-related question to analyze AI accuracy and bias.")
feedback = st.text_area("✍️ **Provide Feedback:**", help="Share any corrections, concerns, or additional insights.")
feedback_submitted = st.button("📩 Submit Feedback")

if feedback_submitted:
    if feedback:
        if save_feedback(query, "LLM response here", feedback):
            st.success("✅ Thank you! Your feedback has been recorded.")
    else:
        st.warning("⚠️ Please enter feedback before submitting.")
        
if st.button("🔎 Analyze Response"):
    if query:
        with st.spinner("Fetching response and analyzing..."):
            llm_response = get_llm_response(query)
            medical_sources = get_medical_sources(query)
            similarity_score, citations = check_similarity(llm_response, medical_sources)
            bias_detected, highlighted_response = detect_bias(llm_response)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("🤖 LLM Response:")
                st.markdown(highlighted_response, unsafe_allow_html=True)
            
            with col2:
                st.subheader("📊 Trustworthiness Score:")
                st.metric(label="Similarity to trusted sources", value=f"{similarity_score:.2f}")
                
                st.subheader("📜 Citations:")
                if citations:
                    for citation in citations:
                        st.markdown(citation)
                else:
                    st.warning("No relevant citations found.")
                
                st.subheader("⚠️ Bias Detection:")
                if bias_detected:
                    bias_report = "Potential bias detected in: " + ", ".join(bias_detected.keys())
                    st.warning(bias_report)
                else:
                    st.success("No explicit bias detected.")
    else:
        st.warning("Please enter a medical question.")
