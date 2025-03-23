import torch

class Trainer:
    def __init__(self, model, dataloader, trainer_config):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=trainer_config.trainer_config["lr"])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.epochs = trainer_config.trainer_config["epochs"]
    
    def train(self):
        for epoch in range(self.epochs):
            for i, (data, target) in enumerate(self.dataloader, 1):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print(f"\rEpoch {epoch}, Loss {loss.item()}", end='')
        print()

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.val_dataloader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print(f"Accuracy: {correct/total}, {correct=}, {total=}")
