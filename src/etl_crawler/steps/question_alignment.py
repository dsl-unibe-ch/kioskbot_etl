CHUNK_QUESTION_ALIGNMENT_PROMPT = """


"""

def chunk_question_alignment(chunk_text: str, doc_question_list: list[str]) -> list[str]:
    """
    Align the chunk text with the document questions.
    """
    