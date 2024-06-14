import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import ViolenceDataset
from model import CNNClassifier, ResNetClassifier, ViTClassifier
from training import Trainer
import pandas as pd

# Define hyperparameters and settings
batch_size = 32
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
image_paths=pd.read_csv("path_to_data_csv", header=None, index= None)
pathline = image_paths[0].to.list()
dataset = ViolenceDataset(image_paths=pathline, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()
input_dim = 224

# Initialize models with adaptive input_dim
cnn_classifier = CNNClassifier(input_dim=input_dim).to(device)
resnet_classifier = ResNetClassifier(input_dim=input_dim).to(device)
vit_classifier = ViTClassifier(input_dim=input_dim).to(device)

# Initialize optimizers and schedulers
cnn_optimizer = optim.Adam(cnn_classifier.parameters(), lr=0.001)
resnet_optimizer = optim.Adam(resnet_classifier.parameters(), lr=0.0005)
vit_optimizer = optim.Adam(vit_classifier.parameters(), lr=0.0003)

cnn_scheduler = optim.lr_scheduler.StepLR(cnn_optimizer, step_size=5, gamma=0.1)
resnet_scheduler = optim.lr_scheduler.CosineAnnealingLR(resnet_optimizer, T_max=num_epochs)
vit_scheduler = optim.lr_scheduler.LambdaLR(vit_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

# Initialize trainer and train models
classifiers = [cnn_classifier, resnet_classifier, vit_classifier]
optimizers = [cnn_optimizer, resnet_optimizer, vit_optimizer]
schedulers = [cnn_scheduler, resnet_scheduler, vit_scheduler]

trainer = Trainer(classifiers, dataloader, criterion, optimizers, schedulers, device=device)
best_classifier = trainer.train(num_epochs=num_epochs)

# Save the best classifier
torch.save(best_classifier.state_dict(), f'{best_classifier.__class__.__name__}_best_classifier.pth')
