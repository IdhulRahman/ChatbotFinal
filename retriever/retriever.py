import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import re

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Default model initialization
DEFAULT_MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(DEFAULT_MODEL_NAME)

def search_vector_store(query, vector_store_path, k=3, return_as_list=False, model_name=None):
    """
    Search FAISS index for the most relevant contexts.
    """
    global model
    if model_name and model_name != DEFAULT_MODEL_NAME:
        logging.info(f"Using custom model: {model_name}")
        model = SentenceTransformer(model_name)

    if not os.path.exists(vector_store_path):
        return "Error: Vector store file not found."
    
    metadata_path = vector_store_path.replace("faiss_index", "metadata.npy")
    if not os.path.exists(metadata_path):
        return "Error: Metadata file not found."

    index = faiss.read_index(vector_store_path)
    texts = np.load(metadata_path, allow_pickle=True)

    query_embedding = model.encode(query, convert_to_numpy=True)
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

    distances, indices = index.search(query_embedding, k)  # Ambil k hasil dari FAISS
    # Ambil indeks unik dan valid
    unique_indices = list(dict.fromkeys([i for i in indices[0] if i >= 0]))
    relevant_texts = [texts[i] for i in unique_indices if i < len(texts)]

    # Batasi hasil ke k
    relevant_texts = relevant_texts[:k]  # Pastikan hanya mengembalikan k hasil

    logging.info(f"Found {len(relevant_texts)} relevant texts.")
    if not relevant_texts:
        return "No relevant data found."

    return relevant_texts if return_as_list else " ".join(relevant_texts)

def is_library_related(query):
    """
    Check if a query is related to the library domain.
    :param query: User query.
    :return: True if related, False otherwise.
    """
    library_keywords = [
    # English
    "book", "author", "publisher", "ISBN", "publication", 
    "library", "recommend", "rating", "pages", "genre", 
    # Indonesian
    "buku", "penulis", "penerbit", "ISBN", "publikasi", 
    "perpustakaan", "rekomendasi", "penilaian", "halaman", "genre",   
    # Spanish
    "libro", "autor", "editor", "ISBN", "publicación", 
    "biblioteca", "recomendar", "calificación", "páginas", "género",
    # French
    "livre", "auteur", "éditeur", "ISBN", "publication", 
    "bibliothèque", "recommander", "évaluation", "pages", "genre"
    ]

    query_lower = query.lower()
    return any(keyword in query_lower for keyword in library_keywords)

def extract_number_from_query(query):
    """
    Extract the number of suggestions requested from the query.
    Defaults to 3 if no number is mentioned.
    :param query: User's query string.
    :return: Extracted number or default (3).
    """
    text_number_map = {
        "satu": 1,
        "dua": 2,
        "tiga": 3,
        "empat": 4,
        "lima": 5,
        "enam": 6,
        "tujuh": 7,
        "delapan": 8,
        "sembilan": 9,
        "sepuluh": 10
    }

    # Search for numbers in the query (both numeric and text)
    numeric_match = re.search(r'\b\d+\b', query)  # Match numbers like "3", "5", etc.
    if numeric_match:
        return int(numeric_match.group())

    for word, num in text_number_map.items():  # Match text numbers like "dua", "tiga"
        if word in query.lower():
            return num

    return 3  # Default if no number found

def format_recommendations(recommendations):
    """
    Format the book recommendations into a structured and readable format.
    """
    formatted = []
    for i, rec in enumerate(recommendations, 1):
        details = rec.split(";")
        title_author = details[0].split(" oleh ") if len(details) > 0 else ["", ""]
        book_info = {
            "Name": title_author[0].strip() if len(title_author) > 0 else "",
            "Authors": title_author[1].strip() if len(title_author) > 1 else "",
            "ISBN": details[1].replace("ISBN ", "").strip() if len(details) > 1 else "",
            "Rating": details[2].replace("rating ", "").strip() if len(details) > 2 else "",
            "PublishYear": details[3].replace("diterbitkan ", "").strip() if len(details) > 3 else "",
            "pagesNumber": details[4].replace("jumlah ", "").strip() if len(details) > 4 else "",
        }

        # Append formatted book info
        formatted.append(
            f"{i}. **{book_info['Name']}**\n"
            f"   \t- **Penulis:** {book_info['Authors']}\n"
            f"   \t- **ISBN:** {book_info['ISBN']}\n"
            f"   \t- **Rating:** {book_info['Rating']}\n"
            f"   \t- **Tahun Terbit:** {book_info['PublishYear']}\n"
            f"   \t- **Jumlah Halaman:** {book_info['pagesNumber']}\n"
        )
    
    return "\n".join(formatted)
