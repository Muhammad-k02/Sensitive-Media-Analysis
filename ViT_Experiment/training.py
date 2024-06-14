from evaluation import Evaluator

class Trainer:
    def __init__(self, classifiers, dataloader, criterion, optimizers, schedulers=None, device='cpu'):
        self.classifiers = classifiers
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizers = optimizers
        self.schedulers = schedulers if schedulers else [None] * len(classifiers)
        self.device = device

    def train(self, num_epochs=10):
        best_classifier = None
        best_accuracy = 0.0

        for classifier, optimizer, scheduler in zip(self.classifiers, self.optimizers, self.schedulers):
            classifier.to(self.device)
            classifier.train()

            for epoch in range(num_epochs):
                running_loss = 0.0
                for inputs, labels in self.dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = classifier(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(self.dataloader.dataset)
                print(f"Classifier: {classifier.__class__.__name__}, Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

                if scheduler:
                    scheduler.step()

            accuracy = self.evaluate_classifier(classifier)
            print(f"Classifier: {classifier.__class__.__name__}, Final Accuracy: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = classifier

        print("Training complete!")
        return best_classifier

    def evaluate_classifier(self, classifier):
        evaluator = Evaluator([classifier], self.dataloader, self.criterion, self.device)
        best_classifier, best_accuracy = evaluator.evaluate_best_classifier()
        return best_accuracy
