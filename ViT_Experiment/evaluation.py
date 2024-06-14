import torch

class Evaluator:
    def __init__(self, classifiers, dataloader, device='cuda'):
        self.classifiers = classifiers
        self.dataloader = dataloader
        self.device = device

    def evaluate_best_classifier(self):
        best_classifier = None
        best_accuracy = 0.0

        for classifier in self.classifiers:
            classifier.to(self.device)
            classifier.eval()

            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in self.dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = classifier(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = classifier

        print(f"Best Classifier: {best_classifier.__class__.__name__}")
        print(f"Best Accuracy: {best_accuracy:.2f}")

        return best_classifier, best_accuracy
