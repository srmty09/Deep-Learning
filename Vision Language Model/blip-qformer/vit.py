import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel
from PIL import Image

image = Image.open("/home/smruti/Desktop/git repos/Deep-Learning/vlm-pipeline/cute-curious-gray-and-white-kitten-in-a-long-shot-photo.jpg")
print(f"Image opened, Image Size: {image.size}")

model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)
print(f"model loaded successfully")

processed_img = processor(images=image,return_tensors="pt")
# print(processed_img)
print(f"processed image size: {processed_img.pixel_values.shape}")

# now pass the processed_img through the model
with torch.no_grad():
    outputs = model(**processed_img)

# print(outputs)

print("Expected: (1,197,768)")
print(f"Embedding shape: {outputs.last_hidden_state.shape}")


