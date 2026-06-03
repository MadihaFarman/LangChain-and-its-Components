from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# embedding = HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-MiniLM-L6-v2')
embedding = HuggingFaceEndpointEmbeddings(
    model= 'sentence-transformers/all-MiniLM-L6-v2',
    task='feature-extraction'
)

documents = [
"Babar Azam (Pakistan): Known for his elegant cover drives, Babar is one of the most consistent batters in modern cricket.",
"Virat Kohli (India): A master of run chases, Kohli's passion and aggression define his cricketing style.",
"Shaheen Afridi (Pakistan): With deadly pace and swing, Shaheen is a nightmare for top-order batters.",
"Rohit Sharma (India): Nicknamed the Hitman, Rohit holds the record for the highest individual ODI score.",
"Shadab Khan (Pakistan): A dynamic all-rounder, Shadab adds energy both with the ball and in the field."
]

query = 'Tell me about madiha.'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

score = cosine_similarity([query_embedding], doc_embeddings)[0]  #Cosine similarity ranges from -1 (opposite) to 1 (identical), with 0 meaning unrelated.
#                                                                  #Give me the 0th row (first row) of the 2D array. """cosine_similarity([query_embedding], doc_embeddings):

# The function expects 2D arrays, so we wrap query_embedding in [].

# Returns a 2D array like: [[0.67, 0.21, 0.89, 0.55, 0.42]], where each number is the similarity between the query and one document.

# [0]: Since the result is a 2D array with just 1 row (your query compared to all documents), this pulls that single row: """
# Convert scores to regular floats and round to 4 decimal places
clean_scores = [(i, round(float(s), 4)) for i, s in enumerate(score)]  #For each index i and value s in the list score,
                                                                       #convert s to a float, round it to 4 decimal places,
                                                                       #and make a tuple (i, rounded_score),
                                                                       #then collect all such tuples into a list and assign it to clean_scores.

index , score = sorted(clean_scores,key=lambda x:x[1])[-1]             #print(sorted(clean_scores,key=lambda x:x[1])[-1])
                                                                       #Sort the list clean_scores by the second value (similarity score) 
                                                                       #in each tuple, and extract the last item from the sorted list 
                                                                       # (which is the highest score). Then unpack that tuple into index and score.

                                                                        #key=lambda x: x[1]
                                                                        #This tells sorted() how to sort the items
                                                                        # lambda is an anonymous function (i.e., a small, unnamed function).
                                                                        #x is each element (tuple) from the list.
                                                                        #x[1] means the second item in the tuple, which is the similarity score.

print(query) 
print(documents[index])
print('Similarity score is : ',score)

#print(list(enumerate(score)))