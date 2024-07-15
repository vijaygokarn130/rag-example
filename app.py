import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import faiss
import numpy as np
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration

# Load models and tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
generator_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
generator = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Prepare documents
documents = [
    {"title": "Document 1", "text": "This is the text of document 1."},
    {"title": "Document 2", "text": "This is the text of document 2."},
    # Add more documents
]
# Index documents
context_embeddings = []
for doc in documents:
    inputs = context_tokenizer(doc['text'], return_tensors='pt')
    embeddings = context_encoder(**inputs).pooler_output.detach().numpy()
    context_embeddings.append(embeddings[0])
context_embeddings = np.array(context_embeddings)
index = faiss.IndexFlatL2(context_embeddings.shape[1])
index.add(context_embeddings)

# Retrieve documents
def retrieve_documents(query, top_k=5):
    inputs = question_tokenizer(query, return_tensors='pt')
    question_embedding = question_encoder(**inputs).pooler_output.detach().numpy()
    _, indices = index.search(question_embedding, top_k)
    return [documents[idx] for idx in indices[0]]

# Generate response
def generate_response(query, retrieved_docs):
    context = " ".join([doc['text'] for doc in retrieved_docs])
    inputs = generator_tokenizer(query + " " + context, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = generator.generate(inputs['input_ids'], num_beams=4, max_length=512, early_stopping=True)
    return generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example usage
query = "What is the text of document 2?"
retrieved_docs = retrieve_documents(query)
print("tthis is retrived docs",retrieve_documents)
response = generate_response(query, retrieved_docs)
print(response)
