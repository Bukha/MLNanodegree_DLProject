# Importing all the needed packages


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import argparse
from os.path import isdir


#####################################
######## Parser for command-line options ##############
#####################################

def get_args():
    parser = argparse.ArgumentParser(description="Training Deep Learning Model")
    parser.add_argument('--save_dir', default='ImageClassifier/', type=str, help="directory for saving checkpoints")
    parser.add_argument('--arch', help='Deep pretrained NN architecture, options: vgg11, vgg13, vgg16, vgg19')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--input_units', default=25088, type=int, help='number of neurons in input layer')
    parser.add_argument('--hidden_units', default=600, type=int, help='number of neurons in hidden layer')
    parser.add_argument('--output_units', default=256, type=int, help='number of neurons in output layer')
    parser.add_argument('--dropout', default=0.2, type=int, help='dropout')
    parser.add_argument('--epochs', default=5, type=int, help='epochs for training')
    parser.add_argument('--gpu', default=True, action='store_true', help='using GPU?')
    return parser.parse_args()


#####################################
######## GPU usage ##############
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


#####################################
######## transforming the train data ##############
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
######## Transforming and loading the training data ##############
#####################################

def train_transformer(train_dir):
    print(train_dir)
    train_transforms= transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    return trainloader

#####################################
######## Transforing and loading the training data ##############
#####################################

def test_transformer(test_dir):
    test_transforms= transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    test_data = datasets.ImageFolder(test_dir ,transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    return testloader

#####################################
######## Choosing from the pre trained models ##############
#####################################
def PreTrainedLoader(arch):
    if type(arch) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".format(arch))
        print(arch)
        model.name = arch
    
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    return model

#####################################
######## Building the desgired Classifier ##############
#####################################
def DesgiredClassifier(input_units, hidden_units,output_units,dropout):

    
    classifier= nn.Sequential(nn.Linear(input_units, hidden_units),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_units, output_units),
                                nn.LogSoftmax(dim=1))
    return classifier
    
#####################################
######## Validating the trained model ##############
#####################################

def Evaluation(loader ,model,criterion,device):
    
    loss = 0
    accuracy = 0
    
    for inputs, labels in loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model.forward(inputs)
        
        batchloss = criterion(logps, labels)
        loss += batchloss.item()
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return accuracy , loss



#####################################
######## The main training fuction ##############
#####################################

def TrainingNN(trainloader,validloader,model,optimizer,criterion,epochs,gpu,print_every,steps):
    print("Training Start.....\n")
    running_loss=0
    device = usedGPU(gpu)
    model.to(device)
    for epoch in range(epochs):
        for inputs, labels in trainloader:        
            steps += 1
        
        # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:

                model.eval()
                with torch.no_grad():
                    accuracy , validloss = Evaluation(validloader,model,criterion,device)
                
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {validloss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
            
    print("\nTraining End")
    return model
#####################################
######## Saving the trained model ##############
#####################################

def savecheckpoint(model, traindata,save_dir):
    if isdir(save_dir):
        
        checkpoint = {'architecture': model.name,
                          'classifier': model.classifier,
                          'class_to_idx': traindata.class_to_idx,
                          'state_dict': model.state_dict()}
        torch.save(checkpoint, save_dir+'checkpoint.pth')
        print ("The model is save in ",save_dir)
        
#####################################
######## Main function ##############
#####################################
def main ():
    args = get_args()
    data_dir = 'ImageClassifier/flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    trainloader  = train_transformer(train_dir)
    testloader  = test_transformer(test_dir)
    validloader  = test_transformer(valid_dir)
    model = PreTrainedLoader(arch=args.arch)
  
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = DesgiredClassifier(input_units=args.input_units
                                          , hidden_units=args.hidden_units
                                          ,output_units=args.output_units
                                          ,dropout=args.dropout)
 
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    print_every = 30
    steps = 0
    trained_model =TrainingNN(trainloader,validloader,model,optimizer,criterion,args.epochs,args.gpu,print_every,steps)
    savecheckpoint(trained_model,Traindata(train_dir),args.save_dir)
        
        

if __name__ == '__main__':
    main()