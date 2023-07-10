from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor, CLIPImageProcessor, CLIPVisionModel
from models import ViTModelWithoutLN

##########################################
# Settings
##########################################
# use_data = 'paired'
use_data = 'categorized'
# use_data = 'caltech-256'
# use_data = 'imagenet'
# use_data = 'phase'

use_model = 'vit-base-patch16-224-in21k'
# use_model = 'clip-vit-base-patch16'

print(use_data)
print(use_model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##########################################
# Define the dataset path and model name
##########################################
if use_data == 'paired':
    dataset_path = '../Datasets/paired/'
elif use_data == 'categorized':
    dataset_path = '../Datasets/categorized/'
elif use_data == 'caltech-256':
    dataset_path = '../Datasets/Caltech-256/'
elif use_data == 'imagenet':
    dataset_path = '../Datasets/ImageNet/'
elif use_data == 'phase':
    dataset_path = '../Datasets/phase/'
else:
    raise NotImplementedError

if use_model.startswith('vit'):
    model_name = 'google/' + use_model
elif use_model.startswith('clip'):
    model_name = 'openai/' + use_model
else:
    raise NotImplementedError

##########################################
# Load dataset and model
##########################################
if use_model.startswith('vit'):
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModelWithoutLN.from_pretrained(model_name).to(device)
elif use_model.startswith('clip'):
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name).to(device)
else:
    raise NotImplementedError


def transform(image):
    return processor(image, return_tensors='pt')['pixel_values']


dataset = ImageFolder(dataset_path, transform=transform)


def collate_fn(examples):
    inputs = torch.concatenate([exam[0] for exam in examples])
    return inputs


dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=64, num_workers=4)

##########################################
# Extract features
##########################################
features = []
with torch.no_grad():
    for inputs in tqdm(dataloader):
        outputs = model(inputs.to(device))

        if use_model.startswith('vit'):
            outputs = outputs[:, 0, :]
        elif use_model.startswith('clip'):
            outputs = outputs.last_hidden_state[:, 0, :]

        features.append(outputs.cpu().numpy())

file_names = np.array([item[0] for item in dataset.samples])
features = np.vstack(features)

##########################################
# Save results
##########################################
np.savez('./temp/%s_%s.npz' % (use_data, use_model), file_names=file_names, features=features)
