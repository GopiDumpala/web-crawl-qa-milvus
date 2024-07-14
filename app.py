# src/app.py

import streamlit as st
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from elasticsearch import Elasticsearch
import numpy as np

# Load the processed data
with open('../data/embeddings_data.json', 'r') as f:
    data = json.load(f)

embeddings = np.array(data['embeddings'])
urls = data['urls']
contents = data['contents']

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
qa_pipeline = pipeline("question-answering")

# Connect to Elasticsearch
es = Elasticsearch()

# Index documents in Elasticsearch for BM25 retrieval
actions = [
    {
        "_index": "cuda_docs",
        "_source": {
            "url": url,
            "content": content
        }
    }
    for url, content in zip(urls, contents)
]
from elasticsearch.helpers import bulk
bulk(es, actions)

def bm25_retrieve(query, top_k=10):
    results = es.search(index="cuda_docs", body={"query": {"match": {"content": query}}}, size=top_k)
    return [hit["_source"]["content"] for hit in results["hits"]["hits"]]

def hybrid_retrieve(query, top_k=10):
    bm25_docs = bm25_retrieve(query, top_k)
    query_embedding = model.encode([query])[0]
    # FAISS search equivalent with Milvus
    D, I = collection.search([query_embedding], anns_field="embedding", param={"metric_type": "L2", "params": {"nprobe": 10}}, limit=top_k)
    bert_docs = [contents[i.id] for i in I[0]]
    combined_docs = bm25_docs + bert_docs
    return combined_docs

st.title("CUDA Documentation Search")

query = st.text_input("Enter your query:")
if query:
    retrieved_docs = hybrid_retrieve(query)
    pairs = [(query, doc) for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
    context = " ".join(sorted_docs[:5])
    result = qa_pipeline(question=query, context=context)
    st.write(f"Q: {query}\nA: {result['answer']}")
