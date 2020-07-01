
# imports lib


import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json
import argparse





#####################################
######## Parser for command-line options ##############
#####################################


def get_args():
    parser = argparse.ArgumentParser(description="Predict Deep Learning Model")
    parser.add_argument('--imageinput',required=True, type=str, help="image link")
    parser.add_argument('--checkpoint',required=True, type=str, help='pre-trained model path')
    parser.add_argument('--categories', default='ImageClassifier/cat_to_name.json', type=str, help='The category file')
    parser.add_argument('--topk', default=5, type=int, help='top_k results')
    parser.add_argument('--gpu', default=False, action='store_true', help='Using GPU?')
    return parser.parse_args()

#####################################
######## flag for GPU or CPU #################
#####################################

def usedGPU (flagGPU):
    device = None
    if flagGPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU...")
    elif flagGPU == False:
        device = torch.device("cpu")
        print("using CPU...")
    else:
        device = torch.device("cpu")
        print("GPU is not found ,using CPU...")
    return device

##################################################
######### Processing the Image as needed #######################
##################################################

def process_image(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    testimage = Image.open(img)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    transimage = transform(testimage)
    
    array_transimage = np.array(transimage)
    
    return array_transimage

##################################################
############# Predicting the top class probabilites #####################
##################################################

def predict(img, model, topk,gpu,train_data,categories):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = usedGPU(gpu)
    processedimage = process_image(img)
    tensorredimage = torch.from_numpy(processedimage).type(torch.FloatTensor)
    tensorredimage = tensorredimage.unsqueeze_(0)
    
    tensorredimage.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(tensorredimage)
        probs = torch.exp(outputs)
        top_p, top_class = probs.topk(topk, dim=1)
        top_p = top_p.numpy()[0]
        top_class = top_class.numpy()[0]   
    idx_to_class = {val: key for key, val in train_data.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]
    if categories != '':
        with open(categories, 'r') as f:
            cat_to_name = json.load(f)
        top_class = [cat_to_name[x] for x in top_class]
    
        

    return top_p, top_class

#####################################
###### transforming the train data ###################
#####################################
def Traindata(dic):
    train_transforms= transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(dic, transform=train_transforms)
    return train_data
#####################################
#########loading the model ##################
#####################################

def load_checkpoint(filepath,train_data):
    """
    Loads deep learning model checkpoint.
    """
    # Load the saved file
    checkpoint = torch.load(filepath)
    
    # Download pretrained model
    model = models.vgg16(pretrained=True);
    
    # Load stuff from checkpoint
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    train_data.class_to_idx = checkpoint['class_to_idx']


    
    return model

#####################################
######### Main function###################
#####################################

def main ():

    
    args = get_args()
    train_data = Traindata('ImageClassifier/flowers/train')
    print ("loading the model..")
    
    model = load_checkpoint(args.checkpoint,train_data)

    probs, classes = predict(args.imageinput, model, args.topk,args.gpu,train_data,args.categories)
    print(f"Top {args.topk} predictions: {list(zip(classes, probs))}")
    
    
    
if __name__ == '__main__':
    main()