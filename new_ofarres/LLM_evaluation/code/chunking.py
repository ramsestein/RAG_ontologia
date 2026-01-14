import spacy

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

def create_smart_chunks(text, max_chars=1200, overlap_sentences=1):
    """
    Splits text by sentences. 
    overlap_sentences=1: Includes the last sentence of the previous chunk 
    in the new chunk to ensure context is not lost (deduplication handled later).
    **For this specific 'Tagging' task where we concatenate strings back, 
    overlapping is tricky. It is safer to NOT overlap but ensure we break 
    ONLY at sentence endings.**
    """
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in doc.sents:
        sent_len = len(sent.text)
        
        # 1200 chars is approx 300-400 tokens. Safe for GPT-4/DeepSeek context.
        if current_len + sent_len > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sent.text]
            current_len = sent_len
        else:
            current_chunk.append(sent.text)
            current_len += sent_len
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks