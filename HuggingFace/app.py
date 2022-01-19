import sys
import os

import matplotlib.pyplot as plt
import PIL
from PIL import Image
import json

import torch
import torchvision
import torchvision.transforms as T

from timm import create_model
import gradio as gr

# Test Different Models
# model_name = 'convnext_xlarge_in22k'
model_name = "convnext_small_1k"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Create ConvNeXt model
model = create_model(model_name, pretrained=True).to(device)

# Define transforms for test
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
NORMALIZE_STD = IMAGENET_DEFAULT_STD
SIZE = 256

# Here we resize smaller edge to 256, no center cropping
transforms = [
              T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
              T.ToTensor(),
              T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
              ]

transforms = T.Compose(transforms)

os.system("wget https://dl.fbaipublicfiles.com/convnext/label_to_words.json")
imagenet_labels = json.load(open('label_to_words.json'))

def inference(img):
    img_tensor = transforms(img).unsqueeze(0).to(device)
    # inference
    output = torch.softmax(model(img_tensor), dim=1)
    top5 = torch.topk(output, k=5)
    top5_prob = top5.values[0]
    top5_indices = top5.indices[0]
    
    result = {}

    for i in range(5):
        labels = imagenet_labels[str(int(top5_indices[i]))]
        prob = float(top5_prob[i])
        result[labels] = prob
    
    return result

inputs = gr.inputs.Image(type='pil')
outputs = gr.outputs.Label(type="confidences",num_top_classes=5)

title = "ConvNeXt"

description = "Gradio demo for ConvNeXt for image classification. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2201.03545' target='_blank'>A ConvNet for the 2020s</a> | <a href='https://github.com/facebookresearch/ConvNeXt' target='_blank'>Github Repo</a></p>"

examples = ['test.jpeg']

gr.Interface(inference, inputs, outputs, title=title, description=description, article=article, analytics_enabled=False, examples=examples).launch(enable_queue=True)
