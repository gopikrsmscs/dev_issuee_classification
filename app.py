
import streamlit as st 
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('all-mpnet-base-v2')

st.title("iSeBetter : Semantic Transformer")
st.header("Analyzing Patterns in Text")


text_input = st.text_area("Enter the issue details below:")


if st.button("Analyse the Issues"):
    # Perform analysis (your existing code)
    query_embedding = embedder.encode(text_input, convert_to_tensor=True)
    corpus_embeddings = torch.load('saved_corpus.pt')
    corpus_embeddings_name = torch.load('saved_corpus_list.txt')
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)

    # Results presentation
    st.subheader("Top 5 Matched Results:")
    result_table = "<table><tr><th>Matched Text</th><th>Score</th></tr>"
    for score, idx in zip(top_results[0], top_results[1]):
        result_table += f"<tr><td>{corpus_embeddings_name[idx]}</td><td>{score:.4f}</td></tr>"
    result_table += "</table>"
    st.markdown(result_table, unsafe_allow_html=True)