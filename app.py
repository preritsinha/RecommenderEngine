from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def recommend_products(products):
    data = pd.read_csv("Products.csv")
    data = data.fillna('NA')
    data['tags'] = data['tags'].apply(lambda x: x.replace(';', ' '))
    data['content'] = data['name']+data['tags']+data['type']+data['description']
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['content'])
    # Get indices of the products in the list
    indices = [data[data['name'] == product].index[0] for product in products]
    # Compute the combined TF-IDF vector (e.g., by averaging)
    combined_vector = np.mean(tfidf_matrix[indices], axis=0)
    # Convert the combined vector to a NumPy array
    combined_vector = np.asarray(combined_vector)
    # Compute cosine similarity with the combined vector
    cosine_sim = linear_kernel(combined_vector, tfidf_matrix)
    # Get similarity scores and sort them
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Exclude the input products from the recommendations
    sim_scores = [score for score in sim_scores if score[0] not in indices]
    # Get top 10 recommendations
    sim_scores = sim_scores[:10]
    product_indices = [i[0] for i in sim_scores]
    return list(set(data['name'].iloc[product_indices]))

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status="Commerce Innovators are awake!")

@app.route('/recommend-items', methods=['POST'])
def recommend_items():
    data = request.get_json()
    products = ast.literal_eval(data.get("products"))
    recommended_products = recommend_products(products)
    return jsonify({"recommended_products": recommended_products})

if __name__ == '__main__':
    app.run(debug=True,port=8080)