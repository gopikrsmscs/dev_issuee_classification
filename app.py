
import streamlit as st 
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('all-mpnet-base-v2')
st.title("iSeBetter : Semantic Transfomer.") 
  
message = st.text_area("Issue details in the text area.") 
  
if st.button("Analyse the Patterns"):
    query_embedding = embedder.encode(message, convert_to_tensor=True)
    corpus_embeddings = torch.load('saved_corpus.pt')
    corpus_embeddings_name = torch.load('saved_corpus_list.txt')
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)

    st.write("Top 5 Matched")
    
    for score, idx in zip(top_results[0], top_results[1]):
         st.write(corpus_embeddings_name[idx], "(Score: {:.4f})".format(score))
    