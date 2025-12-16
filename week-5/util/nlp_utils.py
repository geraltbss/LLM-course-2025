from stqdm import stqdm
import re
from sentence_transformers import util
import torch

def sentencize(pages_and_texts: list[dict], nlp):
    for item in stqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)

        # Make sure all sentences are strings
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]

        # Count the sentences
        item["page_sentence_count_spacy"] = len(item["sentences"])

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# Perform sentence chunking
def semantic_chunker(
    sentences,
    embedder,
    similarity_threshold=0.75,
    min_sentences=1
):
    """
    Groups sentences into chunks based on semantic similarity.
    A new chunk is started when similarity drops below threshold.
    """

    if len(sentences) == 0:
        return []

    # Handle single sentence case
    if len(sentences) == 1:
        return [sentences[0]]

    # Encode sentences
    embeddings = embedder.encode(
        sentences,
        convert_to_tensor=True,
        show_progress_bar=False
    )

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()

        # If similarity is below threshold and we have enough sentences, start new chunk
        if sim < similarity_threshold and len(current_chunk) >= min_sentences:
            # Save the current chunk
            chunks.append(" ".join(current_chunk))
            # Start new chunk with current sentence
            current_chunk = [sentences[i]]
        else:
            # Add sentence to current chunk
            current_chunk.append(sentences[i])

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks