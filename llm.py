"""
Question-answering module.
Uses a distilled BERT model fine-tuned on SQuAD for efficient QA.
"""
from transformers import pipeline

qa_model = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)


def generate_answer(question: str, context_chunks: list, top_k: int = 3) -> str:
    """
    Generate an answer to a question based on provided context chunks.
    
    Args:
        question: The question to answer
        context_chunks: List of relevant text chunks
        top_k: Number of top chunks to use for context
        
    Returns:
        The generated answer
    """
    context_text = " ".join(context_chunks[:top_k])
    
    if not context_text.strip():
        return "No relevant content found to answer this question."
    
    try:
        result = qa_model(question=question, context=context_text)
        answer = result.get("answer", "Unable to extract answer.")
        return answer
        
    except Exception as e:
        print(f"Error during QA generation: {e}")
        return "An error occurred while processing your question. Please try again."
