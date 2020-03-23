#Question 1

import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_dataset

#Loading Dataset
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('iris')

#Compute the sigmoid value of a linear model with weights w
def sigmoid(w,x_i):
    return 1/(1+np.exp(-1*np.dot(w.transpose(),x_i)))

#Compute the gradient of the sigmoid function from above given a datapoint
def sigmoid_grad(weights,x,y):
    rval=np.zeros((1,len(x[0])))
    for i in range(0,len(x)):
        rval=rval+(y[i]-sigmoid(weights,x[i]))*x[i]
    return rval.transpose()

#The BCE loss or negative log likelihood has to be minimized during training
def BCELoss(w,x,y):
    rval=0
    for i in range(0,len(x)):
        rval=rval+y[i]*np.log(sigmoid(w,x[i]))+(1-y[i])*np.log(1-sigmoid(w,x[i]))
    return -1*rval

def Classification_Acc(prediction,actual):
    #Compute classification accuracy given prediction vector and actual vector.
    acc=0
    for i in range(0,np.shape(prediction)[0]):
        if np.array_equal(prediction[i],actual[i]):
            acc=acc+1
    return acc/len(prediction)
#Make model predict on the test set.
def test_pred(w,x_test):
    pred=np.zeros((len(x_test),1))
    for i in range(0,len(x_test)):
        pred[i]=int(sigmoid(w,x_test[i])/0.5)
    return pred
#Get the log likelihood of a set of predictions.
def LogLikelihood(w,x,y):
    rval=0
    for i in range(0,len(x)):
        rval=rval+y[i]*np.log(sigmoid(w,x[i]))+(1-y[i])*np.log(1-sigmoid(w,x[i]))
    return rval
'''Code used specifically for implementing full-batch GD.'''
    
#Minimize the BCELoss by using Full-Batch GD. The converged variable can be set by the 
#user to make the value of teh gradient smaller than a threshold. Returns the set of optimized
#weights, training and validation losses per iteration, and a list of iterations 
def FullBatchGD(x_train,y_train,x_val,y_val,lr,init_values,converged):
    rval=init_values
    iters=[]
    losses=[]
    val_losses=[]
    accum=0
    while(np.linalg.norm(sigmoid_grad(rval,x_train,y_train))>converged):
        iters=iters+[accum]
        accum=accum+1
        losses=losses+[BCELoss(rval,x_train,y_train)]
        val_losses=tes_losses+[BCELoss(rval,x_val,y_val)]
        rval=rval+lr*sigmoid_grad(rval,x_train,y_train)  
    return (rval,losses,iters,val_losses)

'''Code used specifically for implementing SGD.'''
def SGD(x_train,y_train,lr,init_values):
    #shuffling dataset
    np.random.seed(10)
    rng_state = np.random.get_state()
    np.random.shuffle(xtrain)
    np.random.set_state(rng_state)
    np.random.shuffle(ytrain)
    
    rval=init_values
    iters=[]
    losses=[]
    accum=0
    for i in range(0,35000):
        rval=rval+lr*SGD_SigmoidGrad(rval,x_train[i%(len(xtrain)-1)],y_train[i%(len(xtrain)-1)])
        iters=iters+[accum]
        accum=accum+1
        losses=losses+[BCELoss(rval,x_train,y_train)]
        #print(np.linalg.norm(sigmoid_grad(rval,x_train,y_train)))
    return (rval,losses,iters)
    
def SGD_SigmoidGrad(weights,x,y):
    return ((y-sigmoid(weights,x))*x)

#Driver Code for Full Batch GD.
#Preparing the iris dataset

xtest=np.hstack((np.ones((len(xtest),1)),xtest))
ytest=1*ytest[:,(1,)]
xtrain=np.hstack((np.ones((len(xtrain),1)),xtrain))
xvalid=np.hstack((np.ones((len(xvalid),1)),xvalid))
xtrain=np.vstack((xtrain,xvalid))
ytrain=1*ytrain[:,(1,)]
yvalid=1*yvalid[:,(1,)]
ytrain=np.vstack((ytrain,yvalid))
#Initializing Weights
init_weights=np.zeros((len(xtrain[0]),1))

#Getting results for different learning rates. The training phase has been assumed to 
#converge when the magnitude of the gradient is less than 0.01. This section should take 
#fairly long to run.
w1,loss1,iter1,tl1=FullBatchGD(xtrain,ytrain,xtest,ytest,0.001,init_weights,0.01)
w2,loss2,iter2,tl2=FullBatchGD(xtrain,ytrain,xtest,ytest,0.01,init_weights,0.01)
w3,loss3,iter3,tl3=FullBatchGD(xtrain,ytrain,xtest,ytest,0.005,init_weights,0.01)
w4,loss4,iter4,tl4=FullBatchGD(xtrain,ytrain,xtest,ytest,0.0007,init_weights,0.01)

#Plot results.
plt.plot(np.array(iter1),np.array(loss1),label="LR=0.001")
plt.plot(np.array(iter2),np.array(loss2),label="LR=0.01")
plt.plot(np.array(iter3),np.array(loss3),label="LR=0.005")
plt.plot(np.array(iter4),np.array(loss4),label="LR=0.0007")
plt.title("BCE Loss vs Number of Iterations for GD over Iris")
plt.xlabel("Iterations")
plt.ylabel("BCE Loss")
plt.legend()

#Compare all the trained models from above on the test set in terms of loss and accuracy.
print(Classification_Acc(test_pred(w1,xtest),1*ytest[:,(1,)]))
print(Classification_Acc(test_pred(w2,xtest),1*ytest[:,(1,)]))
print(Classification_Acc(test_pred(w3,xtest),1*ytest[:,(1,)]))
print(Classification_Acc(test_pred(w3,xtest),1*ytest[:,(1,)]))
print(LogLikelihood(w1,xtest,1*ytest[:,(1,)]))
print(LogLikelihood(w2,xtest,1*ytest[:,(1,)]))
print(LogLikelihood(w3,xtest,1*ytest[:,(1,)]))
print(LogLikelihood(w4,xtest,1*ytest[:,(1,)]))

#Driver code for SGD

#Reset Dataset
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('iris')
xtest=np.hstack((np.ones((len(xtest),1)),xtest))
ytest=1*ytest[:,(1,)]
xtrain=np.hstack((np.ones((len(xtrain),1)),xtrain))
xvalid=np.hstack((np.ones((len(xvalid),1)),xvalid))
xtrain=np.vstack((xtrain,xvalid))
ytrain=1*ytrain[:,(1,)]
yvalid=1*yvalid[:,(1,)]
ytrain=np.vstack((ytrain,yvalid))

#Initializing Weights
init_weights=np.zeros((len(xtrain[0]),1))
#Getting results for different learning rates. The training phase has been assumed to 
#converge when 35000 training iterations are completed. This section should take 
#fairly long to run.
w1,loss1,iter1=SGD(xtrain,ytrain,0.001,init_weights)
w2,loss2,iter2=SGD(xtrain,ytrain,0.01,init_weights)
w3,loss3,iter3=SGD(xtrain,ytrain,0.005,init_weights)
w4,loss4,iter4=SGD(xtrain,ytrain,0.0007,init_weights)

#Plot Results.
plt.plot(np.array(iter1),np.array(loss1),label="LR=0.001")
plt.plot(np.array(iter2),np.array(loss2),label="LR=0.01")
plt.plot(np.array(iter3),np.array(loss3),label="LR=0.005")
plt.plot(np.array(iter4),np.array(loss4),label="LR=0.0007")
plt.title("BCE Loss vs Number of Iterations for SGD over Iris")
plt.xlabel("Iterations")
plt.ylabel("BCE Loss")
plt.legend()

#Compare all the trained models from above on the test set in terms of loss and accuracy.
print(Classification_Acc(test_pred(w1,xtest),1*ytest[:,(1,)]))
print(Classification_Acc(test_pred(w2,xtest),1*ytest[:,(1,)]))
print(Classification_Acc(test_pred(w3,xtest),1*ytest[:,(1,)]))
print(Classification_Acc(test_pred(w3,xtest),1*ytest[:,(1,)]))
print(LogLikelihood(w1,xtest,1*ytest[:,(1,)]))
print(LogLikelihood(w2,xtest,1*ytest[:,(1,)]))
print(LogLikelihood(w3,xtest,1*ytest[:,(1,)]))
print(LogLikelihood(w4,xtest,1*ytest[:,(1,)]))

#Question 2
import autograd.numpy as np
from autograd import value_and_grad
from data_utils import load_dataset
import matplotlib.pyplot as plt
from data_utils import plot_digit

#Loading Dataset
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('mnist_small')

#Computing the forward pass of the network. The forward pass consists of a linear layer
#followed by a ReLu activation, followed by another linear layer and Relu. The network ends
#with a softmax activation applied on th final linear layer.
def forward_pass(X,W1,W2,W3,b1,b2,b3):
    H1=np.maximum(0, np.dot(X, W1.T) + b1.T) 
    H2=np.maximum(0, np.dot(H1, W2.T) + b2.T)
    linear=np.dot(H2, W3.T) + b3.T 
    a=(np.amax(linear,axis=1)).reshape(len(X),1)
    z=a+(np.log(np.sum(np.exp(linear-a),axis=1))).reshape(len(X),1)
    rval=linear-z
    return rval
   
#Defined previously and used here for SGD.
def BCE_Loss(X,W1,W2,W3,b1,b2,b3,Y):
    prediction=forward_pass(X,W1, W2, W3, b1, b2, b3)
    #print(prediction)
    return -np.sum(np.multiply(prediction,Y))
#Make a prediction over the test set given the network
def test_prediction(x_test,W1,W2,W3,b1,b2,b3):
    prediction=forward_pass(x_test,W1, W2, W3, b1, b2, b3)
    for i in prediction:
        i[np.argmax(i,axis=0)]=True
        i[0:np.argmax(i,axis=0)]=False
        i[np.argmax(i,axis=0)+1:len(i)]=False
    return prediction

#Compute the testing accuracy of a classification problem.
def test_accuracy(prediction,actual):
    acc=0
    for i in range(0,np.shape(prediction)[0]):
        if np.array_equal(prediction[i],actual[i]):
            acc=acc+1
    return acc/len(prediction)

#Determines the inputs for which the network was unsure:
#the maximum confidence associated to any class<50%
def unsure_network(x_test,W1,W2,W3,b1,b2,b3):
    prediction=forward_pass(x_test,W1, W2, W3, b1, b2, b3)
    idx=[]
    print(prediction[399])
    for i in range(0,len(prediction)):
        if(prediction[i][np.argmax(prediction[i],axis=0)]<np.log(0.5)):
            idx=idx+[(i,np.exp(np.amax(prediction[i],axis=0)))]
    return idx

    
#Using the autograd function to compute the gradient of the loss function
#with respect to the weights and biases of the network.
gradient=value_and_grad(BCE_Loss,argnum=[1,2,3,4,5,6])

#Initializing weights using the Xavier Scheme.
w1 = np.random.randn(100,784)/np.sqrt(784)
w2 = np.random.randn(100,100)/np.sqrt(784)
w3 = np.random.randn(10,100)/np.sqrt(784)
#Initialize Biases to zero
B1 = np.zeros((100,1)) 
B2 = np.zeros((100,1)) 
B3 = np.zeros((10,1)) 

#Carrying out Stochastic Gradient Descent over the network using a mini-bacth size of 250
#and customizable learning rates. Saving the training and validation losses at each iteration.
losses=[]
iters=[]
val_losses=[]
for i in range (0,1000):
    if((250*(i+1))%10000==0):
        (loss, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad))=gradient(xtrain[(250*i)%10000:10000],
                                                                   w1, w2, w3, B1, B2, B3,ytrain[(250*i)%10000:10000])
    else:
        (loss, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad))=gradient(xtrain[(250*i)%10000:(250*(i+1))%10000],
                                                                w1, w2, w3, B1, B2, B3,ytrain[(250*i)%10000:(250*(i+1))%10000])
    val_losses=val_losses+[BCE_Loss(xvalid,w1,w2,w3,B1,B2,B3,yvalid)]
    if(loss==0):
        print(i)
    losses=losses+[loss]
    iters=iters+[i]
    #print(W3_grad)
    w1=w1-0.001*W1_grad
    w2=w2-0.001*W2_grad
    w3=w3-0.001*W3_grad
    B1=B1-0.001*b1_grad
    B2=B2-0.001*b2_grad
    B3=B3-0.001*b3_grad
    
#Driver code for Question 2.

#Plotting training and validation data.
plt.plot(np.array(iters),np.array(losses),label='Training Loss')
plt.plot(np.array(iters),np.array(val_losses),label='Validation Loss')
plt.title('Training and Validation Losses vs Iterations (lr=0.00001)')
plt.xlabel('Iterations')
plt.ylabel('BCE Loss')
plt.legend()

#Determining test accuracy.
prediction=test_prediction(xtest,w1,w2,w3,B1,B2,B3)
print(test_accuracy(prediction,ytest))
#Determine all the inputs over which the network is unsure.
print(unsure_network(xtest,w1,w2,w3,B1,B2,B3))
