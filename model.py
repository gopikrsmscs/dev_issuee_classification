import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import load_dataset

dataset = load_dataset("gopikrsmscs/torch-issues")


# Create InputExamples from your dataset
examples = []
for i, row in dataset['train'].iterrows():
    title = row['Title']
    body = row['Body']
    examples.append(InputExample(texts=[title, body]))

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Define a DataLoader for training
train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)

# Fine-tune the model
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,  # You can adjust the number of training epochs
    warmup_steps=100,
    optimizer_params={'lr': 1e-4},
)

# Save the fine-tuned model
model.save('iSeBetter')
