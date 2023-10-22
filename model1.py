from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import load_dataset
import pinecone

embedder = SentenceTransformer('all-mpnet-base-v2')

dataset = load_dataset("gopikrsmscs/torch-issues")

columns = dataset['train'].column_names
columns_to_keep = ["Title", "Body"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
dataset = dataset.remove_columns(columns_to_remove)

dataset_dict = dataset['train']

examples = []
for row in dataset_dict:
    title = row['Title']
    #body = row['Body']
    examples.append(title)

file_paths = 'saved_corpus.txt'
torch.save(examples[0:1000],file_paths)
corpus_embeddings = embedder.encode(examples[0:1000], convert_to_tensor=True)
file_path = 'saved_corpus.pt'

# Save the tensor to the file
torch.save(corpus_embeddings, file_path)
