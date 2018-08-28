# Imports here
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torchvision as torchvision
import torch as torch
from torchvision import models
import json

# Import classifier generator
from classifier import Network

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
    
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Load Checkpoint
    checkpoint = torch.load(in_arg.checkpoint)
    model = models[checkpoint['model_name']]

    model.classifier = load_checkpoint(checkpoint)

    # Predict Class
    probs, classes = predict(in_arg.input, model, proc_name, in_arg.top_k)
    
    # Convert data & Retrieve Class names
    probs = probs.cpu().numpy()
    filterd_names = { classNo: cat_to_name[str(model.classifier.class_to_idx[str(classNo)])] for classNo in classes.cpu().numpy() }
    nameList = list(filterd_names.values())
    
    df = pd.DataFrame({
        "name" : pd.Series(nameList),
        "probability" : pd.Series(probs)
    })
    
    # Print Results
    print("Predicted Flower Name: " + nameList[0])
    print("Probability: " + str(probs[0]))
    print("Top " + str(in_arg.top_k) + " classes are:")
    print(df.head())
    
# Function - Load Checkpoint
def load_checkpoint(checkpoint):
    model = Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Download and load the test data
    size = 256, 256
    
    # load the image
    img_pil = Image.open(image)
    
    # Resize to 256
    img_pil.thumbnail(size)
    
    # Crop 224 x 224
    width, height = img_pil.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    img_pil = img_pil.crop((left, top, right, bottom))
    
    # Convert Color Channels (RGB to Binary)
    np_image = np.array(img_pil)
    np_image_bin = np_image / 255
    
    # Normalize Color Channels
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image_norm = (np_image_bin - mean)/std
    
    np_image_final = np.transpose(np_image_norm, (2,0,1))
    
    return np_image_final

def predict(image_path, model, proc_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # GPU or CPU?
    model.to(proc_name)
    model.eval()
    
    img_tensor = torch.from_numpy(process_image(image_path))
    
    img_tensor = img_tensor.to(proc_name)
    
    img_tensor.unsqueeze_(0)

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img_tensor.float())
    
    return torch.exp(output).data[0].topk(topk)

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
    parser.add_argument('input', type=str,
                        help='path to input image file')
    parser.add_argument('checkpoint', type=str,
                        help='path to checkpoint file')
    parser.add_argument('--top_k', type=int, default=3, 
                        help='number of top probability classes')
    parser.add_argument('--category_names', type=str, default='../aipnd-project/cat_to_name.json', 
                        help='JSON with category names mapped to indices')
    parser.add_argument('--gpu', action='store_true',
                        help='GPU or CPU?')

    # returns parsed argument collection
    return parser.parse_args()

# Call to main function to run the program
if __name__ == "__main__":
    main()