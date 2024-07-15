# process_data.py

import json
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Load scraped data
with open('../data/cuda_docs.json', 'r') as f:
    data = json.load(f)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Chunking and creating embeddings
embeddings = []
urls = []
contents = []

for entry in data:
    sentences = nltk.sent_tokenize(' '.join(entry['content']))
    sentence_embeddings = model.encode(sentences)
    embeddings.extend(sentence_embeddings)
    urls.extend([entry['url']] * len(sentences))
    contents.extend(sentences)

embeddings = np.array(embeddings)

# Connect to Milvus
connections.connect()

# Define Milvus schema
fields = [
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1]),
    FieldSchema(name="url", dtype=DataType.STRING),
    FieldSchema(name="content", dtype=DataType.STRING)
]
schema = CollectionSchema(fields, "cuda_doc_embeddings")

# Create collection
collection = Collection("cuda_docs", schema)
collection.insert([embeddings, urls, contents])

# Create index
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)
collection.load()

# Save the collection for local use
embeddings_data = {'embeddings': embeddings.tolist(), 'urls': urls, 'contents': contents}
with open('../data/embeddings_data.json', 'w') as f:
    json.dump(embeddings_data, f)
