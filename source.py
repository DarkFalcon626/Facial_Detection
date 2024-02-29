# -*- coding: utf-8 -*-
"""
Source.py
------------------------------------------------------
Source code file for classes and functions for facial
recognition.
------------------------------------------------------
Created on Sat Feb 10 15:55:24 2024
@author: Andrew Francey
"""

from PIL import Image
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import os
import json

## ---------------------------------------------------------------------
## Functions
## ---------------------------------------------------------------------

def process(img, res):
    '''
    Consumes an image im and crops it into a square then rescales to the 
    resolution res.

    Parameters
    ----------
    im : 
        DESCRIPTION.
    res : TYPE
        DESCRIPTION.

    Returns
    -------
    Image object
        A square image of resolution res.
    '''
    
    img = Image.open(img)
    width, height = img.size # Get the dimensions of the image.
    
    ## If the image is not a square crop the image into a square.
    if width < height:
        left = 0
        right = width
        top = (height - width)/2 # Set the new height to the width
        bottom = (height + width)/2
        img = img.crop((left, top, right, bottom))
        
    elif width > height:
        left = (width - height)/2 # Set the new width to the height
        right = (width + height)/2
        bottom = height
        top = 0
        img = img.crop((left, top, right, bottom))
    
    img = img.resize((res,res)) # Rescale the image to the choosen resolution.

    return img


def conv_width(w,p,k,s):
    '''
    Determine the outputed width of a convolation layer.

    Parameters
    ----------
    w : Int
        Inputted width.
    p : Int
        The padding around the array.
    k : Int
        Width of the kernal.
    s : Int
        The number of the stride.

    Returns
    -------
    Int
        The width of the output of a convolation layer.
    '''
    return 1 + (w + (2*p) - k)//s
    

def pool_width(w, p, k, s):
    '''
    Determine the outputed width of a pooling layer.

    Parameters
    ----------
    w : Int
        Inputted width.
    p : Int
        The padding around the array.
    k : Int
        Width of the kernal.
    s : Int
        The number of the stride.

    Returns
    -------
    Int
        The width of the output of a pooling layer.
    '''
    return 1+(w+(2*p)-(k-1)-1)//s

##---------------------------------------------------------------------------
## Classes
## --------------------------------------------------------------------------

class Data():
    '''
    Class for processing the data into tensors and converting all the data to 
    the same shape and size. Then split the data into test and training groups.
    
    Parameters
    ----------
    data_path : Str
        File location for the data set file.
    param : JSON file
        Hyperparameters for the data set, such as the resolution of the final 
        data and the percentage of the data to use for the test group.
    device : torch.device
        The device to store the torch tensors on. Such as cpu or cuda.
    batch : Bool
        If True create the data in a batchable formate. Default is True.
    
    Attributes
    ----------
    n_test : Int
        The number of data points in the test set.
    n_train : Int
        The number of data points in the training set.
    res : Int
        The width of the image.
    test_data : Torch Tensor
        Tensor with the image data for testing.
    test_target : Torch Tensor
        Tensor with the target vectors for testing the model.
    train_data : Lstof Numpy Arrays (Only if Batch = True)
        A list of the training data points.
    train_data : Torch Tensor (Only if Batch = False)
        Tensor with the image data for training.
    train_targets : Torch Tensor (Only if Batch = False)
        Tensor with the target vectors for traing the model.
        
    Functions
    ---------
    create_batch(self, param)
        Creates randomized batchs from the training data. creates both a 
        batch_data and batch_target attributes for the data class.
    '''
    
    def __init__(self, data_path, param, device, batch=True):
        
        
        self.res = int(param['resolution'])
        
        ## Set up list to store the processed data
        data = []
        targets = []
        
        i = 0 # Set an indicator for the target.
        for folder in ['face', 'nonface']: # Cycle through both a face or no face.
            path = data_path +'\\' +folder # Get the location of the images.
            for img in os.listdir(path):
                img = process(path+'\\'+img, self.res) # Process the image.
                
                ## Convert image to an array.
                img_array = np.asarray(img)
                img_array = img_array.transpose()
                
                if img_array.shape[0] == 3:
                    # Add the image array and the corresponding target to a list.
                    data.append(img_array)
                    targets.append(i)
              
            ## Switch to the next target value. 
            i += 1       
        
        ## Shuffle the list to mix up the face and no face data.
        data, targets = shuffle(data, targets)
        
        ## Determine the size of the test and training sets.
        self.n_test = int(len(data)*float(param['test_size']))
        self.n_train = int(len(data) - self.n_test)
        
        ## Create the test set.
        test_data = data[:self.n_test]
        test_targets = targets[:self.n_test]
        
        ## Convert to an array
        test_data = np.array(test_data)
        test_targets = np.array(test_targets)
        
        ## Convert to tensors.
        self.test_data = torch.tensor(test_data).to(device)
        test_targets = torch.tensor(test_targets)
        
        ## Convert to targets to vectors.
        self.test_targets = func.one_hot(test_targets.to(torch.int64), 
                                        num_classes=2).to(device)
        
        ## Next set up the training data
        if batch:
            ## If batching leave the data in a list for creating batchs from.
            self.train_data = data[self.n_test:]
            self.train_targets = targets[self.n_test:]
        else:
            ## Else convert the data to data and target tensors.
            train_data = data[self.n_test:]
            train_targets = targets[self.n_test:]
            
            ## Convert to array
            train_data = np.array(train_data)
            train_targets = np.array(train_targets)
            
            ## Convert to tensor
            self.train_data = torch.tensor(train_data).to(device)
            train_targets = torch.tensor(train_targets)
            ## Convert targets to vectors.
            self.train_targets = func.one_hot(train_targets.to(torch.int64), 
                                              num_classes=2).to(device)

    def create_batch(self, num_batchs, device):
        '''
        Creates randomized batch to use for training.
        
        Parameters
        ----------
        num_batchs : Int
            The number of batchs to create from training data.
        
        Effects
        -------
        Create the following attributes.
        batched_data : Torch Tensor
            Batched data for training.
        batched_targets : Torch Tensor
            The corresponding target vectors for the batchs.
        device : torch.device
            The device to store the torch tensors on. Such as cpu or cuda.
            
        Returns
        -------
        None.
        '''
        
        ## Determine the size of each batch
        size_batch = self.n_train//num_batchs
        
        ## Shuffle the data around each time this function is called allowing
        ##  for random batchs each time.
        data, targets = shuffle(self.train_data, self.train_targets)
        
        ## Create lists to store the batchs data and targets.
        batched_data = []
        batched_targets = []
        
        ## Split the data into batchs of size_batch.
        for i in range(0, self.n_train, size_batch):
            batch_data = data[i:i+size_batch]
            batch_targets = targets[i:i+size_batch]
            batch_data = np.array(batch_data)
            batch_targets = np.array(batch_targets)
            if len(batch_data) == size_batch:
                batched_data.append(batch_data)
                batched_targets.append(batch_targets)
        
        ## Transfor to array
        batched_data = np.array(batched_data)
        batched_targets = np.array(batched_targets)
        
        ## Transform into tensors.
        self.batched_data = torch.tensor(batched_data).to(device)
        batched_targets = torch.tensor(batched_targets)
        
        ## Turn the targets to vectors.
        self.batched_targets = func.one_hot(batched_targets.to(torch.int64),
                                            num_classes=2).to(device)


class Net(nn.Module):
    '''
    An 8 layer CNN for facial recagnition. using 6 Convolation filters, 2 Max
    pooling Functions and 2 fully connected layers with ReLU activation
    functions at each layer.
    Uses the torch.nn.Module as a parent class.

    Parameters
    ----------
    data : Data object
        The Data object containing the test and training data.
    net_params : JSON file
        Json file with networks hyperparamets.

    Attributes
    ----------
    layer1 : Sequential
        A convolation layer with ReLU activation followed by a 2D max pooling
        and a 2D batch normalization.
    layer2 : Sequential
        A convolation layer with ReLU activation followed by a 2D max pooling
        and a 2D batch normalization.
    layer3 : Sequential
        A convolation layer with ReLU activation.
    layer4 : Sequential
        A convolation layer with ReLU activation.
    layer5 : Sequential
        A convolation layer with ReLU activation followed by a 2D max pooling
        and a 2D batch normalization.
    layer6 : Squential
        A fully connected linear layer with a ReLU activation.
    layer7 : Squential
        A fully connected linear layer with a ReLU activation.
    layer8 : Squential
        A fully connected linear layer.
        
    Functions
    ---------
    forward(self, x)
        Runs a data group x throught the network outputting a tensor with the
        prediction of the network.
    backprop(self, inputs, targets, loss, optimizer)
        Runs a data group inputs through the network comparing to the target 
        group targets using the loss function loss. Then using an optimizer to
        update the weights of the network.
    test(self, data, loss)
        Runs a data group through the network and comparing its data and 
        targets using a loss function. test does not update any weights.
    '''
    
    def __init__(self, data, net_params):
        super(Net, self).__init__()
        
        ## Retreving the width, kernal, pool, stride and channels.
        w1 = data.res
        k1 = net_params['kernal_1']
        p1 = net_params['padding_1']
        s1 = net_params['stride_1']
        conv1 = net_params['conv1_out_channels']
        
        pw1 = conv_width(w1, p1, k1, s1)
        pk1 = net_params['pool_kernal_1']
        pp1 = net_params['pool_padding_1']
        ps1 = net_params['pool_stride_1']
        
        w2 = pool_width(pw1,pp1,pk1,ps1)
        k2 = net_params['kernal_2']
        p2 = net_params['padding_2']
        s2 = net_params['stride_2']
        conv2 = net_params['conv2_out_channels']
        
        pw2 = conv_width(w2, p2, k2, s2)
        pk2 = net_params['pool_kernal_2']
        pp2 = net_params['pool_padding_2']
        ps2 = net_params['pool_stride_2']
        
        w3 = pool_width(pw2, pp2, pk2, ps2)
        k3 = net_params['kernal_3']
        p3 = net_params['padding_3']
        s3 = net_params['stride_3']
        conv3 = net_params['conv3_out_channels']
        
        w4 = conv_width(w3, p3, k3, s3)
        k4 = net_params['kernal_4']
        p4 = net_params['padding_4']
        s4 = net_params['stride_4']
        conv4 = net_params['conv4_out_channels']
        
        w5 = conv_width(w4, p4, k4, s4)
        k5 = net_params['kernal_5']
        p5 = net_params['padding_5']
        s5 = net_params['stride_5']
        conv5 = net_params['conv5_out_channels']
        
        pw5 = conv_width(w5, p5, k5, s5)
        pk5 = net_params['pool_kernal_5']
        pp5 = net_params['pool_padding_5']
        ps5 = net_params['pool_stride_5']
        
        w6 = pool_width(pw5, pp5, pk5, ps5)
        fc1 = net_params['fc1']
        fc2 = net_params['fc2']
        
        ## Build the network.
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, conv1, k1, s1, p1),
            nn.ReLU(),
            nn.MaxPool2d(pk1,ps1,pp1),
            nn.BatchNorm2d(conv1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(conv1,conv2,k2,s2,p2),
            nn.ReLU(),
            nn.MaxPool2d(pk2,ps2,pp2),
            nn.BatchNorm2d(conv2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(conv2,conv3,k3,s3,p3),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(conv3, conv4, k4,s4,p4),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(conv4, conv5, k5,s5,p5),
            nn.ReLU(),
            nn.MaxPool2d(pk5,ps5,pp5),
            nn.BatchNorm2d(conv5))
        self.layer6 = nn.Linear((w6**2)*conv5, fc1)
        self.layer7 = nn.Linear(fc1,fc2)
        self.layer8 = nn.Linear(fc2, 2)
    
    def forward(self,x):
        '''
        Passes a torch tensor x of the data through the models network and 
        returns a torch tensro with the models guesses.

        Parameters
        ----------
        x : Torch Tensor
            Data to be evaluated by the model.

        Returns
        -------
        Torch Tensor
            The models best guess at the classification.
        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.size(0),-1) # Flatten the feature maps to a linear layer.
        x = func.relu(self.layer6(x))
        x = func.relu(self.layer7(x))
        x = self.layer8(x)
        
        return func.softmax(x,dim=(1))
    
    def backprop(self, inputs, targets, loss, optimizer):
        '''
        Passes a data set inputs through the model then using a loss function
        loss to compare the models output to the target targets. Then update the
        weights of the model using the optimizer.

        Parameters
        ----------
        inputs : Torch Tensor
            The data set to be evaluated by the model.
        targets : Torch Tensor
            The data sets target.
        loss : Torch Loss Function
            The loss function to use in determining the loss value.
        optimizer : Torch Optimizer
            The optimization function to determine how to update the weights.

        Returns
        -------
        Float
            The loss value of the training.
        '''
        ## Set the model to train mode.
        self.train()
        
        inputs = inputs.float() # Converts the tensors to torch float points.
        targets = targets.float()
        
        outputs = self.forward(inputs) # Pass the data through the model.
        obj_val = loss(outputs, targets) # Determine the loss value.
        optimizer.zero_grad() # Reset the gradient
        obj_val.backward() # Update the models values
        optimizer.step() # Increment the step counter.
        
        return obj_val.item()
    
    def test(self, data, loss):
        '''
        Passes the data set through the model and returning the loss value using
        the loss function to evaluate the datas inputs and targets.

        Parameters
        ----------
        data : Data object
            The data set.
        loss : Torch Loss Function
            The loss function to use in determining the loss value.

        Returns
        -------
        Float
            The loss value of the testing.
        '''
        
        ## Get the inputs and targets from the test data.
        inputs = data.test_data
        targets = data.test_targets
        
        self.eval() # Set the model to eval mode.
        with torch.no_grad(): # Evaluate the model without the use of gradients.
            inputs = inputs.float() # Converts the tensors to torch float points.
            targets = targets.float()
            
            # Pass the test data through the model and comapare to the targets.
            cross_val = loss(self.forward(inputs), targets)
        
        return cross_val.item()
        
    
##-------------------------------------------------------------------------
## Main code
##-------------------------------------------------------------------------

if __name__ == '__main__':
    
    ## Determine if there is a GPU to use for training.
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    
    device = torch.device(dev)
    
    ## Determine the file path of the file.
    file_location = os.path.dirname(__file__) + '\\'
    
    ## Open the json file for the hyperparameters.
    with open(file_location+'param.json') as paramfile:
        param = json.load(paramfile)
        
    ## Test the data class and processing on small sets
    data = Data(file_location+'test_set',param['Data'], device)
    data2 = Data(file_location+'test_set',param['Data'], device, False)
    
    ## Test setting up the model and feeding it through the model.
    model = Net(data, param['Net'])
    model.to(device)
    
    inputs = data.test_data
    inputs = inputs.float()
    model.forward(inputs)
    
    
    
    
    
    
    
    
    
    