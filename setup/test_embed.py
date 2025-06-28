from transformers import AutoTokenizer, AutoModel
import torch

model_name = "Qwen/Qwen3-Embedding-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Encode a sample sentence
inputs = tokenizer("Hello world", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# This gives you the last hidden state or pooled embedding depending on the model
embedding = outputs.last_hidden_state.mean(dim=1)  # [batch_size, embedding_dim]
print(embedding.shape)


