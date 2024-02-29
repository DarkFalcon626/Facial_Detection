# -*- coding: utf-8 -*-
"""
main.py
--------------------------------------------------
Main code file for training the facial recognition
model.
--------------------------------------------------
Created on Fri Feb 16 15:36:10 2024
@author: Andrew
"""

import torch
import time
import os
import json, argparse
import pickle
import winsound
import torch.nn as nn
import source as src
import numpy as np
import pylab as plt

##----------------------------------------------------------------------
## Functions
##----------------------------------------------------------------------
def prep(param):
    '''
    Process the data set into workable tensors to train and test the model. 
    Creates the CNN network model to be trained.

    Parameters
    ----------
    param : JSON file
        Hyperparamets for the data such as the desired resolution and the
        precentage of the data to use as a test set. Also includes 
        hyperparameters for the model layers.

    Returns
    -------
    data : Data object
        The processed data for the testing and training of the model.
    model : Net object
        The model to be trained and tested.
    '''
    
    data = src.Data(args.data_path, param['Data'], device, args.batch)
    model = src.Net(data, param['Net'])
    model.to(device) # Move the model to a gpu if one is avalible.
    return data, model

def run(param, model, data):
    '''
    Trains and tests the model outputing the loss values at the desired epochs.
    Return the loss values for both the training and testing.

    Parameters
    ----------
    param : JSON file
        Hyperparameters for the training of the model. Such as the learn rate,
        number of epochs and batchs, the momentum, the interval of epochs to
        display, the milestones, and the gamma rate to update the learning rate.
    model : Net object
        The CNN model to train and test.
    data : Data object
        The data sets for testing and training of the model.

    Returns
    -------
    loss_vals : lstof Floats
        The loss values per epoch of training.
    cross_vals : lstof Float
        The loss values per epoch of testing.
    '''
    
    ## Using a stochastic gradient descent optimizer for updating the models 
    ##   values. 
    optimizer = torch.optim.SGD(model.parameters(), lr=param['lr'],
                                momentum=param['momentum'])
    
    ## Using a learn rate scheduler to update the learning rate at set milestones
    ##  by the gamma rate.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     gamma = param['gamma'],
                                                     milestones=[param['mile']])
    
    ## Using the binary cross-entropy loss function to determine the loss values.
    loss = nn.BCELoss(reduction='mean')
    
    ## Create lists to store the training and test losses.
    loss_vals = []
    cross_vals = []
    
    num_epochs = int(param['num_epochs'])
    for epoch in range(num_epochs): # Loop through each epochs.
        if args.batch: # If the training is using epochs.
            num_batchs = int(param['num_batchs'])
            data.create_batch(num_batchs, device) ## Create randomized batchs.
            
            batch_loss = [] # Store the training loss for each batch.
            for batch in range(num_batchs): # Loop over the batchs.
                ## Pass the batch through the model and update the model values
                ##  using the optimizer to minimize the loss value.
                train_val = model.backprop(data.batched_data[batch], 
                                           data.batched_targets[batch],
                                           loss, optimizer)
                batch_loss.append(train_val) # Append the batch loss to the batch list.
            
            ## Take the avarage of the batch losses.
            loss_val = sum(batch_loss)/num_batchs
            loss_vals.append(loss_val) ## Append to the training loss.
        
        else: ## If training is not using batching.
            ## Pass the whole training data set through the model.
            train_val = model.backprop(data.train_data, data.train_targets,
                                       loss, optimizer)
            loss_val.append(train_val)
        
        test_val = model.test(data, loss) # Test the model on the test set.
        cross_vals.append(test_val)
        
        scheduler.step() # Update the scheduler for the milestones.
        
        ## Determine if the loss values should be printed to the screen.
        if epoch == 0:
            print('Epoch [{}/{}] ({:.1f}%)'.format(epoch+1, num_epochs,\
                                               ((epoch + 1)/num_epochs)*100) + \
                  '\tTraining Lass: {:.5f}'.format(train_val) + \
                      '\tTest Loss: {:.5f}'.format(test_val))
            winsound.Beep(1000,100) # Audio que to alert of the update.
        if (epoch+1) % param['display_epochs'] == 0:
            print('Epoch [{}/{}] ({:.1f}%)'.format(epoch+1, num_epochs,\
                                               ((epoch + 1)/num_epochs)*100) + \
                  '\tTraining Lass: {:.5f}'.format(train_val) + \
                      '\tTest Loss: {:.5f}'.format(test_val))
            winsound.Beep(1000,100) # Audio que to alert of the update.
    
    print('Final training Loss: {:.6f}'.format(loss_vals[-1]))
    print('Final test loss: {:.6f}'.format(cross_vals[-1]))
    
    return loss_vals, cross_vals

def save_value(save):
    '''
    Determines if the input is a excepted value to a yes or no question, if not
    get a new input.

    Parameters
    ----------
    save : STR
        Input string to a yes or no question.

    Returns
    -------
    save : Bool
        Awnser to the yes or no question.
    '''
    
    ## Reduce any uppercase to lowercase to maintain the meaning of the word.
    save = save.lower()
    
    ## Determine if the awnser is true or false.
    if save in ['true', '1', 'yes']:
        save = True
    elif save in ['false', '0', 'no']:
        save = False
    else: # If the awnser is not an excepted value ask the question again.
        save = input('Value entered is not a proper response. Please enter \
                     either true, 1, yes or false, 0, no. ->')
        save = save_value(save) # Check the new awnser.
    
    return save


##-------------------------------------------------------------------------
## Main code
##-------------------------------------------------------------------------
      
if __name__ == "__main__":

    start_time = time.time() # Get the start time.
    
    ## Determine if there is a GPU to use for training.
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = "cpu"
    
    device = torch.device(dev)
    
    ## Determine the file path of the file.
    file_location = os.path.dirname(__file__) + '\\'
    
    ## Create arguments that are needed for the training. This arguments can be
    ##  changed from the defaults in the command line.
    parser = argparse.ArgumentParser(description="Training of Facial detection model")
    parser.add_argument('--param', default=file_location+'param.json', 
                        type=str, help='Json file for hyperparameters.')
    parser.add_argument('--data-path', default=file_location+'data_set',
                        type=str, help='Location of test and training images.')
    parser.add_argument('--model-name', default=file_location+'facial_reconition.pkl',
                        type=str, help='Name to save the model as.')
    parser.add_argument('--batch', default=True, 
                        type=bool, help='Boolean value to use batching.')
    parser.add_argument('--fig-name', default=file_location+'loss_fig.png',
                        type=str, help='Name of the image to save the loss plot as.')
    args = parser.parse_args()
    
    ## Open the hyperparameter file.
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    
    print('Beginning processing of data.')
    data, model = prep(param) # Process the data and create the model.
    print('Data has finished processing and the model has been initalized.')
    print('Beginning training...')
    loss_vals, cross_vals = run(param['exec'], model, data) # Train the model.
    
    train_time = time.time()-start_time # Determine the time took to train.
    ## Convert the time into a more readable formate of hours, minutes and seconds.
    if (train_time//3600) > 0:
        hours = train_time//3600
        mins = (train_time-hours*3600)//60
        secs = train_time-hours*3600-mins*60
        print('The model trained in {} hours, {} mins and {:.0f} seconds'.format(hours,mins,secs))
    elif (train_time//60) > 0:
        print('The model trained in {} mins and {:.0f} seconds'.format(train_time//60,
                                                               train_time-(train_time//60)*60))
    else:
        print('The model trained in {:.0f} seconds'.format(train_time))
    
    x = np.arange(1,len(loss_vals)+1) # Axis for the number of epochs.
    
    ## Plot the loss values vs the epoch.
    plt.plot(x,loss_vals, label='Training loss')
    plt.plot(x,cross_vals, label='Test loss')
    plt.title('Loss per Epoch.')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    ## Asks the user if the model should be saved or not.
    save = input('Do you want to save the model and training data. -> ')
    save = save_value(save) 
    
    ## If save the save the model and the figure.
    if save:
        with open(args.model_name, 'wb') as f:
            pickle.dump(model, f) # Save the model as a pickle file.
            f.close()
        plt.savefig(args.fig_name, format='png')
        
        print('Model and plot saved as {} and {}'.format(args.model_name,
                                                         args.fig_name))
        
        
        
