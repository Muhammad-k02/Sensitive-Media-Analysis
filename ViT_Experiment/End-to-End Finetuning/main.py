import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from combination import CombinedModel
from ViT_Experiment.training import Trainer
from ViolenceEndToEndDataset import ViolenceEndToEndDataset

batch_size = 32
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data = pd.read_csv('path_to_csv_file', header=None, index=None)
frame_paths = data[0].tolist()
labels = data[1].tolist()

dataset = ViolenceEndToEndDataset(frame_paths=frame_paths, labels=labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define criterion
criterion = nn.CrossEntropyLoss()


combined_model = CombinedModel(num_classes=2).to(device)
optimizer = optim.Adam(combined_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Initialize trainer and train the model
trainer = Trainer([combined_model], dataloader, criterion, [optimizer], [scheduler], device=device)
best_classifier = trainer.train(num_epochs=num_epochs)

# Save the best classifier
torch.save(best_classifier.state_dict(), 'best_classifier_stage2.pth')
