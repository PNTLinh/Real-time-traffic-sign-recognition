from transformers import BlipForConditionalGeneration, BlipProcessor
import torch
import cv2
import numpy as np
from PIL import Image

class VLMClassifier:
    """ Using VLM to classify images """
    def __init__(self, model_name='Salesforce/blip-image-captioning-base', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """ Args:
            model_name: name of vlm in HuggingFace
        """
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"Loaded VLM model: {model_name} on {self.device}")
        
    def preprocess_image_for_vlm(self, image:np.ndarray) -> Image.Image:
        """ Preprocess image for VLM input
        Args:
            image: input image in numpy array format (H, W, C)
        """
        
    def classify(self, image, prompt="What is in the image?"):
        """ Classify image using VLM
        Args:
            image: input image in numpy array format (H, W, C) or PIL Image
            prompt: text prompt to guide the classification
        Returns:
            caption: generated caption for the image
        """
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption
        