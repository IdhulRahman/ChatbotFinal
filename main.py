import os
import streamlit as st
from datetime import datetime
from database.cache import save_to_cache, get_cached_answer
from database.csv_to_vector import load_csv_data, build_vector_store
from retriever.retriever import search_vector_store, is_library_related, extract_number_from_query, format_recommendations
from utils.make_finetune import save_to_fine_tuning_file

CSV_FILE_PATH = "retriever/data/library.csv"
VECTOR_STORE_PATH = "retriever/data/faiss_index"
USE_CACHED_ANSWERS = 0

def check_and_build_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        st.info("Building vector store from the CSV file...")
        data = load_csv_data(CSV_FILE_PATH)
        build_vector_store(data, VECTOR_STORE_PATH)
        st.success("Vector store created successfully.")
    else:
        st.success("Vector store is already available.")

def main():
    st.title("📚 Library Chatbot")
    st.markdown("Welcome to the **Library Chatbot**! Ask questions about the library, and get personalized book recommendations.")

    check_and_build_vector_store()

    use_cached_answers = bool(USE_CACHED_ANSWERS)

    # Input query from the user
    query = st.text_input("Ask your question (e.g., 'Recommend 3 mystery books'):")

    if query:
        # Check if query is related to the library
        if not is_library_related(query):
            st.warning("Sorry, I cannot provide summaries, resumes, or explain the content of books. "
                       "Please ask about book titles, authors, or recommendations.")
            return

        # Check for cached answer if enabled
        if use_cached_answers:
            cached_answer = get_cached_answer(query)
            if cached_answer:
                st.subheader("Cached Answer:")
                st.write(cached_answer)
                return

        # Extract the number of suggestions from the query
        num_suggestions = extract_number_from_query(query)
        if num_suggestions > 99:
            st.warning("I can only provide up to 99 suggestions. Please ask for fewer recommendations.")
            return
        if num_suggestions <= 0:
            st.warning("Please specify a valid number of suggestions (greater than 0).")
            return

        # Measure response time
        start_time = datetime.now()
        context = search_vector_store(query, VECTOR_STORE_PATH, k=num_suggestions)
        response_time = (datetime.now() - start_time).total_seconds()

        if not context or context.startswith("Error"):
            st.error("No relevant data found in the database.")
            return

        # Format recommendations
        recommendations = context.split(". ")[:num_suggestions]
        formatted_recommendations = format_recommendations(recommendations)

        # Display recommendations and response time
        st.subheader("Recommended Books:")
        st.write(formatted_recommendations)
        st.write(f"⏳ **Response Time:** {response_time:.3f} seconds")

        # Collect user feedback
        st.subheader("Rate This Response:")
        rating = st.radio("How would you rate this response?", ("1: Bad", "2: Neutral", "3: Good"))
        if st.button("Submit Rating"):
            if rating.startswith("2"):
                save_to_cache(query, formatted_recommendations)
                st.success("Response cached.")
            elif rating.startswith("3"):
                save_to_cache(query, formatted_recommendations)
                save_to_fine_tuning_file(query, context, formatted_recommendations)
                st.success("Response cached and saved for fine-tuning.")
            elif rating.startswith("1"):
                st.info("Response discarded based on your feedback.")
    else:
        st.info("Type your question above to start!")

if __name__ == "__main__":
    main()
