import torch
import numpy as np
from transformers import BertTokenizer, BertModel

MAX_EMBEDDING_DIM = 1024

# Padding tensor to MAX_EMBEDDING_DIM with zeros
def pad_tensor(tensor: torch.Tensor, max_len: int = MAX_EMBEDDING_DIM) -> torch.Tensor:
    """Pad tensor with zeros to max_len dimensions."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.ndim == 1 and tensor.shape[0] < max_len:
        tensor = torch.cat((tensor, torch.zeros(max_len - tensor.shape[0], device=tensor.device)), dim=0)
    elif tensor.ndim == 2 and tensor.shape[1] < max_len:
        tensor = torch.cat((tensor, torch.zeros(tensor.shape[0], max_len - tensor.shape[1], device=tensor.device)), dim=1)
    elif tensor.ndim > 2:
        raise ValueError("Invalid tensor shape: Expected 1D or 2D tensor.")
    return tensor

# Text Embedding using Bert Model
def text_embedding(text: str, model_name: str = "bert-base-uncased") -> torch.Tensor:
    """Generate text embedding using a Bert model."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    encoded_input = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return torch.mean(model_output.last_hidden_state, dim=1)  # Mean pooling strategy

# OpenAI Embeddings
def openai_embeddings_tensor(client, texts: list, dimensions: int = 64, model: str = "text-embedding-3-large") -> torch.Tensor:
    """Fetch embeddings from OpenAI's embedding API and convert to tensor."""
    texts = [text.replace("\n", " ") for text in texts]  # Remove newlines to improve embedding quality
    embeddings = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    return torch.as_tensor([emb.embedding for emb in embeddings])
