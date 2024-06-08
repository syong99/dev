# STEP 1
from sentence_transformers import SentenceTransformer

# STEP 2
model = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 3
sentences1 = "The weather is lovely today."
sentences2 = "it's me"

# STEP 4
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

print(embeddings1.shape)
print(embeddings1.shape)

# STEP 5
similarities = model.similarity(embeddings1, embeddings2)
print(similarities)
