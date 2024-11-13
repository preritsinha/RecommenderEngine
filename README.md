
# Product Recommendation API

This is a Flask-based API for generating product recommendations using content similarity. It reads product data from a CSV file and uses TF-IDF vectorization and cosine similarity to recommend similar products based on a list of user-selected items.

## Requirements

- Python 3.x
- Required libraries: Flask, pandas, numpy, scikit-learn

## Installation

1. Clone the repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `Products.csv` is in the root directory, containing the following columns:
   - `name` - Product name
   - `tags` - Tags for the product (separated by semicolons)
   - `type` - Product type
   - `description` - Product description

## Usage

1. Run the Flask server:
   ```bash
   python app.py
   ```

2. API Endpoints:
   - **Health Check:** `GET /health`
     - Response: `{"status": "Commerce Innovators are awake!"}`
   
   - **Recommend Items:** `POST /recommend-items`
     - Request body (JSON):
       ```json
       {
         "products": "['Product1', 'Product2', ...]"
       }
       ```
     - Response:
       ```json
       {
         "recommended_products": ["RecommendedProduct1", "RecommendedProduct2", ...]
       }
       ```

## How It Works

- The `recommend_products` function loads data from `Products.csv`, cleans it, and builds TF-IDF vectors based on product content (name, tags, type, and description).
- The API uses cosine similarity to recommend the top 10 most similar products, excluding the products provided in the input.

## License

This project is open-source and available for personal and commercial use.
