import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import torchvision as torchvision
import torch as torch
from torchvision import datasets, transforms, models
import seaborn as sb

# Import classifier generator
from classifier import Network 

# Define Dataset directories
train_dir = '/train'
valid_dir = '/valid'

# Import Network
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet121 = models.densenet121(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16, 'densenet': densenet121}

def main():
    # Creates & Retrieves CMD Args
    in_arg = get_input_args()
    
    proc_name = 'cuda' if in_arg.gpu else 'cpu'
    model = models[in_arg.arch]
    
    # Classifier Input values (default = densenet, 1024)
    classInputNo = 1024
    
    if in_arg.arch == 'resnet':
        classInputNo = 512
    elif in_arg.arch == 'alexnet':
        classInputNo = 9216
    elif in_arg.arch == 'vgg':
        classInputNo = 25088
    
    # Define transforms for training, validation, and testing sets
    transform_train = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets with ImageFolder
    train_dataset = datasets.ImageFolder(in_arg.data_dir + train_dir, transform=transform_train)
    valid_dataset = datasets.ImageFolder(in_arg.data_dir + valid_dir, transform=transforms_test)
    # test_dataset = datasets.ImageFolder(test_dir, transform=transforms_test)

    # Define dataloaders
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size = 32)
    # dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size = 32)

    # Freeze parameters to prevent backdrop
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = Network(classInputNo, 102, in_arg.hidden_units, drop_p=0.5)

    # Train Network
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)

    epochs = in_arg.epochs
    print_every = 40
    steps = 0

    # change to cuda
    model.to(proc_name)

    print("Training Start...")

    for e in range(epochs):
        model.train()

        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloader_train):
            steps += 1

            inputs, labels = inputs.to(proc_name), labels.to(proc_name)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, dataloader_valid, criterion, proc_name)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(dataloader_valid)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(dataloader_valid)))

                running_loss = 0

                # Make sure training is back on
                model.train()

    print("Training End!")

    # Save checkpoint
    checkpoint = {'input_size': classInputNo,
                  'output_size': 102,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  'state_dict': model.classifier.state_dict(),
                  'class_to_idx': train_dataset.class_to_idx,
                  'model_name': in_arg.arch
                 }

    torch.save(checkpoint, in_arg.save_dir + '/checkpoint.pth')
    
    print("Checkpoint exported in: checkpoint.pth")

# Retrieve CMD args
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse    module. This function returns these arguments as an
     mentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates 3 command line arguments args.dir for path to images files,
    # args.arch which CNN model to use for classification, args.labels path to
    # text file with names of dogs.
    parser.add_argument('data_dir', type=str, default='../aipnd-project/flowers',
                        help='an integer for the accumulator')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='path to folder of images')
    parser.add_argument('--arch', type=str, default='densenet',
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[512],
                        help='hidden layer units')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--gpu', action='store_true',
                        help='text file that has dognames')

    # returns parsed argument collection
    return parser.parse_args()

# Function - Validation
def validation(model, testloader, criterion, proc_name):
    test_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        
        inputs, labels = inputs.to(proc_name), labels.to(proc_name)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

# Call to main function to run the program
if __name__ == "__main__":
    main()