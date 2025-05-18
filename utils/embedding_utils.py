## Model loading for all the embeddings

import torch
from PIL import Image
import requests
import torch
from transformers import AutoModel, AutoProcessor
import numpy as np


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess_images(image_list):
    """
    Preprocess the image list to ensure compatibility with encode_image.

    Args:
        image_list (list): List of images in PIL.Image.Image, file paths, or NumPy arrays.

    Returns:
        list: List of PIL.Image.Image objects.
    """
    processed_images = []
    if not isinstance(image_list, list):
        image_list = [image_list]
    for img in image_list:
        # print(type(img))
        if isinstance(img, np.ndarray):
            # Convert NumPy array to PIL image
            processed_images.append(Image.fromarray(img))

        elif isinstance(img, str) and not (img.startswith("http://") or img.startswith("https://")):
            image = Image.open(img).convert("RGB")
            processed_images.append(image)

        elif isinstance(img, str) and img.startswith("http://") or img.startswith("https://"):
            # If it's a URL, download the image
            response = requests.get(img, stream=True)
            response.raise_for_status()
            processed_images.append(Image.open(response.raw).convert("RGB"))

        
        elif isinstance(img, Image.Image):
            processed_images.append(img)

        else:
            raise ValueError("Images must be PIL.Image.Image or NumPy array.")
    return processed_images

import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
import numpy as np


class JinaClipModel:
    def __init__(self, model_name='jinaai/jina-clip-v2', truncate_dim=512):
        """
        Initialize Jina CLIP model.

        :param model_name: Model name from Hugging Face.
        :param truncate_dim: Truncated embedding dimension.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, truncate_dim=truncate_dim, cache_folder="cache").to(self.device)

    def encode_text(self, text_list, normalize=True):
        """
        Encode a list of text inputs into embeddings.

        :param text_list: List of text strings.
        :param normalize: Whether to normalize embeddings.
        :return: Normalized text embeddings as a NumPy array.
        """
        return self.model.encode(text_list, normalize_embeddings=normalize, device=self.device)

    def encode_images(self, image_list, normalize=True):
        """
        Encode a list of images into embeddings.

        :param image_list: List of PIL.Image.Image, file paths, or URLs.
        :param normalize: Whether to normalize embeddings.
        :return: Normalized image embeddings as a NumPy array.
        """
        processed_images = preprocess_images(image_list)
        return self.model.encode(processed_images, normalize_embeddings=normalize, device=self.device)

    def encode_text_image(self, text, image_input, normalize=True):
        """
        Encode both text and a single image together.

        :param text: Text input.
        :param image_input: Image input (URL, local path, or PIL Image).
        :param normalize: Whether to normalize embeddings.
        :return: Joint text and image embeddings.
        """
        image = preprocess_images([image_input])[0]
        return self.model.encode([text, image], normalize_embeddings=normalize, device=self.device)
