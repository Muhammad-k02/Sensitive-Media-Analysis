import torch
import torch.nn as nn
from transformers import AutoModel

class CombinedModel(nn.Module):
    def __init__(self, saved_classifier):
        super(CombinedModel, self).__init__()
        self.dino = AutoModel.from_pretrained('facebook/dinov2-base')
        self.classifier = saved_classifier

        for param in self.dino.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        with torch.no_grad():
            features = self.dino(**inputs).last_hidden_state.mean(dim=1)
        out = self.classifier(features)
        return out
