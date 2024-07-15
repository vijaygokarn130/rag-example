# rag-example
Retrieval-Augmented Generation. 

Tokenize the query:h 
inputs = question_tokenizer(query, return_tensors='pt')
Compute the query embedding:

Compute the query embedding:
outputs = question_encoder(**inputs)
Extract the pooled output:

Extract the pooled output:
pooled_output = outputs.pooler_output
Detach from the computation graph:

Detach from the computation graph:
detached_output = pooled_output.detach()
Convert to NumPy array:

Convert to NumPy array:
question_embedding = detached_output.numpy()