import torch.cuda

from models.CNN import CNN
from torchsummary import summary
import torch.optim as optim
from loader import getLoader
import torch.nn as nn
def train():
    model = CNN().double()
    if torch.cuda.is_available():
        model = model.cuda()
    #summary(model,(3,512,512))
    trainLoader, valLoader = getLoader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for e in range(2):  # loop over the dataset multiple times
        train_loss = 0.0
        for data, labels in trainLoader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target, labels)
            # Calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss += loss.item()

        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer
        for data, labels in valLoader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target, labels)
            # Calculate Loss
            valid_loss += loss.item()

        print(f'Epoch {e + 1} \t\t Training Loss: { train_loss / len(trainLoader)} \t\t Validation Loss: { valid_loss / len(valLoader)}')

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f\
                }--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss

            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')


    print('Finished Training')
    return model