from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

# Load a pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Prepare your dataset (replace with your dataset loading code)
train_texts = ["Text of issue 1", "Text of issue 2", ...]
labels = [0, 1, ...]  # 0 for non-relevant, 1 for relevant

# Tokenize and convert your dataset to tensors
input_ids = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
labels = torch.tensor(labels)

# Set up data loaders
dataset = torch.utils.data.TensorDataset(input_ids["input_ids"], input_ids["attention_mask"], labels)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Define optimizer and loss
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Fine-tune the model
model.train()
for epoch in range(3):  # Replace with desired number of epochs
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained("/path/to/save/model")

# You can now use this model for semantic search.
