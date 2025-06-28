from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import gc

_embedder = None
X = None
y = None
clf = None

# === Training data ===
examples = [
    ("What is the capital of France?", "agent:retrieve"),
    ("Search for the history of neural networks.", "agent:retrieve"),
    ("What is the product with ", "agent:retrieve"),
    ("Translate this to French: Hello", "tool:translator"),
    ("Get today’s weather and write an email update.", "agent:multi_tool"),
    ("Draft an email using the API results.", "agent:multi_tool"),
    ("Send an email to me with the data", "agent:email"),
    ("Write me a poem", "default:unknown"),
    ("take a photo of this", "default:unknown"),
    ("Draft up an email and include the following data", "agent:email"),
    ("Im looking for a formula with", "agent:retrieve"),
    ("Look for sku number", "agent:retrieve"),
    ("Search for item called", "agent:retrieve"),
]


def load_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # Force CPU
    return _embedder


def train_classifier():
    global X, y, clf
    embedder = load_embedder()
    X = embedder.encode([e[0] for e in examples])
    y = [e[1] for e in examples]
    clf = LogisticRegression().fit(X, y)


def route_query(query, unload_after=False):
    global clf
    if clf is None:
        train_classifier()

    embedder = load_embedder()
    embedding = embedder.encode([query])
    prediction = clf.predict(embedding)[0]
    confidence = clf.predict_proba(embedding)[0].max()

    if unload_after:
        unload_embedder()

    if confidence < 0.4:
        return {
            "route": "default",
            "intent": "unknown",
            "confidence": confidence,
            "raw_input": query,
            "reason": "Low confidence"
        }

    route_parts = prediction.split(":") if ":" in prediction else [prediction, "default"]

    return {
        "route": route_parts[0],
        "intent": route_parts[1],
        "confidence": confidence,
        "raw_input": query
    }


def add_training_example(example_text, label):
    global X, y, clf
    embedder = load_embedder()
    new_embedding = embedder.encode([example_text])
    X = np.vstack([X, new_embedding])
    y.append(label)
    clf = LogisticRegression().fit(X, y)


def unload_embedder():
    global _embedder
    _embedder = None
    gc.collect()


def main():
    # Train the classifier with initial examples
    train_classifier()
    
    # Test queries
    test_queries = [
        "What is the capital of Singapore?",
        "Translate this to Spanish: Good morning",
        "Can you find the SKU for this product?",
        "Email the latest sales report",
        "Generate a recipe",
    ]

    for query in test_queries:
        result = route_query(query)
        print(f"\nQuery: {query}")
        print("Result:")
        for k, v in result.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()