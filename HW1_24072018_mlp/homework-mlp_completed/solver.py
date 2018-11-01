
import numpy as np


def train(model, criterion, optimizer, dataset, max_epoch, batch_size, disp_freq):
    avg_train_loss, avg_train_acc = [], []
    avg_val_loss, avg_val_acc = [], []
    
    max_train_iteration = int(dataset.train.num_examples / batch_size)
    max_val_iteration = int(dataset.validation.num_examples / batch_size)
    for epoch in range(max_epoch):
        batch_train_loss, batch_train_acc = [], []
        batch_val_loss, batch_val_acc = [], []
        for iteration in range(max_train_iteration):
    
            # Get training data and label
            train_x, train_y = dataset.train.next_batch(batch_size, shuffle=True)
            train_x -= np.mean(train_x, axis=1, keepdims=True)
            
            # forward pass
            logit = model.forward(train_x)
            criterion.forward(logit, train_y)
            
            # backward pass
            delta = criterion.backward()
            model.backward(delta)
    
            optimizer.step(model)
    
            batch_train_loss.append(criterion.loss)
            batch_train_acc.append(criterion.acc)
    
            if iteration % disp_freq == 0:
                print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
                    epoch, max_epoch, iteration, max_train_iteration, np.mean(batch_train_loss), np.mean(batch_train_acc)))
    
        for iteration in range(max_val_iteration):
            val_x, val_y = dataset.validation.next_batch(batch_size, shuffle=False)
            val_x -= np.mean(val_x, axis=1, keepdims=True)
    
            logit = model.forward(val_x)
            loss = criterion.forward(logit, val_y)
    
            batch_val_loss.append(criterion.loss)
            batch_val_acc.append(criterion.acc)
    
        avg_train_loss.append(np.mean(batch_train_loss))
        avg_train_acc.append(np.mean(batch_train_acc))
        avg_val_loss.append(np.mean(batch_val_loss))
        avg_val_acc.append(np.mean(batch_val_acc))
    
        print('Epoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
            epoch, avg_train_loss[-1], avg_train_acc[-1]))
        print('Epoch [{}]\t Average validation loss {:.4f}\t Average validation accuracy {:.4f}\n'.format(
            epoch, avg_val_loss[-1], avg_val_acc[-1]))
        
    return model, avg_val_loss, avg_val_acc


def test(model, criterion, dataset, batch_size, disp_freq):
    print("Testing")
    max_test_iteration = int(dataset.test.num_examples / batch_size)
    batch_test_acc = []
    for iteration in range(max_test_iteration):
        test_x, test_y = dataset.test.next_batch(batch_size, shuffle=False)
        test_x -= np.mean(test_x, axis=1, keepdims=True)
    
        logit = model.forward(test_x)
        loss = criterion.forward(logit, test_y)
        batch_test_acc.append(criterion.acc)
    
        if iteration % disp_freq == 0:
            print("Batch [{}][{}]\t Accuracy {:.4f}".format(
                    iteration, max_test_iteration, np.mean(batch_test_acc)))
    
    print("The test accuracy is {:.4f}.\n".format(np.mean(batch_test_acc)))