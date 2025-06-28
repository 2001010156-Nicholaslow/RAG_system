from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

text = "The quick brown fox jumps over the lazy dog."

print("BERT Tokens:", bert_tokenizer.tokenize(text))
print("QWEN Tokens:", qwen_tokenizer.tokenize(text))

embeddings = model.encode(text)
print(embeddings)
print("Embedding shape:", len(embeddings))

'''
========Results==========
BERT Tokens: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
QWEN Tokens: ['The', 'Ġquick', 'Ġbrown', 'Ġfox', 'Ġjumps', 'Ġover', 'Ġthe', 'Ġlazy', 'Ġdog', '.']
[ 0.02413564 -0.04545963 -0.00333167 ...  0.04504577 -0.00728704
 -0.00102431]
Embedding shape: 1024

MisMatch in token. Switch to QWEN tokenizer. Qwen uses Uses BPE (Byte Pair Encoding), BERT uses lowercased tokens.
If we want to use BERT tokenizer, switch to BERT tokenizer and embedding.
'''