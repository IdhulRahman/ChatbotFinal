import os
import csv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_csv_data(csv_file):
    """
    Load data from CSV file and create descriptive strings for each entry.
    Args:
        csv_file (str): Path to the CSV file.
    Returns:
        list: List of descriptive strings for each row in the CSV.
    """
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')  # Adjusted for ';' delimiter
        for row in reader:
            description = (
                f"{row['Name']} oleh {row['Authors']};ISBN {row['ISBN']}; "
                f"rating {row['Rating']};diterbitkan {row['PublishYear']}; "
                f"jumlah {row['pagesNumber']}."
            )
            data.append(description)
    return data

def build_vector_store(csv_file, vector_store_path):
    """
    Build a vector store from CSV data if the FAISS index doesn't already exist.
    Args:
        csv_file (str): Path to the CSV file.
        vector_store_path (str): Path to save the FAISS index.
    """
    if os.path.exists(vector_store_path):
        print("Vector store already exists. Skipping conversion.")
        return

    print("Vector store not found. Converting CSV data to vectors...")
    data = load_csv_data(csv_file)
    embeddings = model.encode(data)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, vector_store_path)

    # Save metadata
    metadata_path = vector_store_path.replace("faiss_index", "metadata.npy")
    with open(metadata_path, "wb") as f:
        np.save(f, np.array(data))

    print("Vector store successfully created.")
