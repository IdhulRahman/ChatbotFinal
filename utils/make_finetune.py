import json

def save_to_fine_tuning_file(query, context, answer, fine_tune_file_path="database/finetuning/fine_tuning_data.jsonl"):
    """
    Save data as JSONL for fine-tuning when rated '3'.
    :param query: User's question.
    :param context: Retrieved context from the vector store.
    :param answer: Generated answer based on the query and context.
    :param fine_tune_file_path: Path to save fine-tuning data.
    """
    # Prepare fine-tuning data in JSONL format
    fine_tune_entry = {
        "instruction": query,      # The user's question
        "context": context,        # The relevant context retrieved
        "response": answer         # The generated response
    }
    
    # Append the data to the fine-tuning file
    with open(fine_tune_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(fine_tune_entry, ensure_ascii=False) + "\n")

    print(f"Saved to fine-tuning file: {fine_tune_file_path}")
