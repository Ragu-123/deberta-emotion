import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple

def load_model_and_tokenizer(repo_id: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Loads a pre-trained model and tokenizer from the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID of the model on the Hugging Face Hub.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device: {device}")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        
        model.to(device)
        model.eval()
        
        print(f"Successfully loaded model '{repo_id}'")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise