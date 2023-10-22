
import tensorflow as tf
import streamlit as st 
from datasets import load_dataset


dataset = load_dataset("gopikrsmscs/torch-issues")
print(dataset['train'])

st.title("SmartCluster") 
  
message = st.text_area("Input the input details in the text area.") 
  
if st.button("Analyse the Patterns"):
    