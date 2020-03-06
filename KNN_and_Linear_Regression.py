##### KNN Regression Code (Question 1) #######

#Imports
import numpy as np
from data_utils import load_dataset
import math
import matplotlib.pyplot as plt

xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('mauna_loa')

def KNN(x_loader,y_loader,point,k,norm):
    prev_dist=float('inf')
    nn=[]
    x_vals=[]
    for j in range (0,k):
        for i in range(0,len(x_loader)):
            if(np.linalg.norm(point-x_loader[i])<prev_dist and list(x_loader[i]) not in x_vals):
                prev_dist=np.linalg.norm(np.subtract(np.array(point),np.array(x_loader[i])),ord=norm)
                x_val=x_loader[i]
                y_val=y_loader[i][0]
        nn=nn+[y_val]    
        x_vals=x_vals+[list(x_val)]
        prev_dist=float('inf')
    return sum(nn)/len(nn)

#Helper function to compute RMSE for cross validation (c1 and c2 are the boundaries designating the 
#validation set)
def RMSE_Helper(x_train,y_train,k,norm,c1,c2):
    acc=0
    for i in range(c1,c2):
        acc=acc+(y_train[i]-KNN(x_train[0:c1]+x_train[c2:len(x_train)],y_train[0:c1]+
                y_train[c2:len(y_train)],x_train[i],k,norm))**2
    return math.sqrt(acc/(c2-c1))

#Function to compute average RMSE for KNN predictions during Cross Validation 
def GetCrossValRMSE(x_train,y_train,k,norm):
    rval=[]
    acc_list=[[int(4*len(x_train)/5),len(x_train)],[int(3*len(x_train)/5),int(4*len(x_train)/5)],
                [int(2*len(x_train)/5),int(3*len(x_train)/5)],[int(1*len(x_train)/5),
                int(2*len(x_train)/5)],[0,int(1*len(x_train)/5)]]
    for i in range (0,len(acc_list)):
        rval=rval+[RMSE_Helper(x_train,y_train,k,norm,acc_list[i][0],acc_list[i][1])]
    return sum(rval)/len(rval)

#Function to determine best K-value for a dataset given a norm. Running this function with different
#norms allows us to determine the best norm to use as well. Results are based on average RMSE during
#cross validation. Returns the best K value anf the minimum cross-validation error.
def BestK_Regression(x_train,y_train,norm):
    error=[]
    k_values=[]
    for k in range (1,20):
        error=error+[GetCrossValRMSE(x_train,y_train,k,norm)]
        k_values=k_values+[k]
        print(k,error[k-1])
    e=np.array(error)
    k=np.array(k_values)
    plt.plot(k,e)
    return (k_values[error.index(min(error))],min(error))

def Plot_Val_Helper(x_train,y_train,k,norm,c1,c2):
    pred=[]
    for i in range(c1,c2):
        pred=pred+[KNN(x_train[0:c1]+x_train[c2:len(x_train)],y_train[0:c1]+
                y_train[c2:len(y_train)],x_train[i],k,norm)]
    return pred

def Plot_Validation(x_train,y_train,k,norm):
    predictions=[]
    cutoff=[[int(4*len(x_train)/5),len(x_train)],[int(3*len(x_train)/5),int(4*len(x_train)/5)],
                [int(2*len(x_train)/5),int(3*len(x_train)/5)],[int(1*len(x_train)/5),
                int(2*len(x_train)/5)],[0,int(1*len(x_train)/5)]]
    cutoff.reverse()
    for i in range (0,len(cutoff)):
        predictions=predictions+Plot_Val_Helper(x_train,y_train,k,norm,cutoff[i][0],cutoff[i][1])
        #print(Plot_Val_Helper(x_train,y_train,k,norm,cutoff[i][0],cutoff[i][1]))
        #print('-------')
    plt.title("Validation Predictions vs Actual Data")
    plt.scatter(x_train[0:len(x_train)],predictions,s=1,label='Predicted')
    plt.scatter(x_train[0:len(x_train)],y_train[0:len(x_train)],s=1,label='Actual')
    plt.legend()
#Calculates the RMSE error on the test set, given the training points, k-value and norm 
def Test_Error(x_train,y_train,x_test,y_test,k,norm):
    results=[]
    for i in range (0,len(x_test)):
        results=results+[(KNN(x_train,y_train,x_test[i],k,2)-y_test[i])**2]
    return math.sqrt(sum(results)/len(results))

#Plots the test predictions of the KNN and the acutal test targets to get a qualtitative perspective 
def Test_Predictions(x_train,y_train,x_test,y_test,k,norm):
    results=[]
    for i in x_test:
        results=results+[KNN(x_train,y_train,i,k,norm)]
    plt.plot(x_test,results,label='Predicted')
    plt.plot(x_test,y_test,label='Actual')
    plt.title('Actual vs Predicted Values (Test Set)')
    plt.xlabel('Test Inputs')
    plt.ylabel('Outputs')
    plt.legend()


#Driver Code

shuffling=True #Set this to False for no shuffling

#This code is used to shuffle the training and validation data. It was noticed that shuffling
#allows for better cross-validation accuracy and led to different optimal K-values and test accuracy.
if shuffling:
    np.random.seed(10)
    rng_state = np.random.get_state()
    np.random.shuffle(xtrain)
    np.random.set_state(rng_state)
    np.random.shuffle(ytrain)
    rng_state = np.random.get_state()
    np.random.shuffle(xvalid)
    np.random.set_state(rng_state)
    np.random.shuffle(yvalid)

#Finding the best K-value for 2-norm
print(BestK_Regression(list(xtrain)+list(xvalid),list(ytrain)+list(yvalid),1))
#Finding test error for best K-Value
print(Test_Error(list(xtrain)+list(xvalid),list(ytrain)+list(yvalid),xtest,ytest,12,2))
#Plotting Validation predictions
Plot_Validation(list(xtrain)+list(xvalid),list(ytrain)+list(yvalid),2,2)
#Plotting test predictions
Test_Predictions(list(xtrain)+list(xvalid),list(ytrain)+list(yvalid),xtest,ytest,12,2)




##### KNN Classification Code (Question 2) #######

#Load Data
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('mnist_small')

#KNN function for classification. Inputs: 'training x values','training y values',query point,K-value 
#and norm. Returns an array of predicted features.
def KNN_Classifier(x_loader,y_loader,point,k,norm):
    prev_dist=float('inf')
    nn=[]
    x_vals=[]
    for j in range (0,k):
        for i in range(0,len(x_loader)):
            if(np.linalg.norm(point-x_loader[i])<prev_dist and list(x_loader[i]) not in x_vals):
                prev_dist=np.linalg.norm(np.subtract(np.array(point),np.array(x_loader[i])),ord=norm)
                x_val=x_loader[i]
                y_val=y_loader[i]
        nn=nn+[y_val]  
        x_vals=x_vals+[list(x_val)]
        prev_dist=float('inf')
    rval=[]
    for i in range(0,len(nn[0])):
        t=0
        f=0
        for j in nn:
            if(j[i]==False):
                f=f+1
            else:
                t=t+1
        if(t>f):
            rval=rval+[True]
        else:
            rval=rval+[False]
    return np.array(rval)

#Function returns accuracy of the KNN algorithm on the validation set (no cross validation involved).
#Value between 0 and 1.
def GetValAcc(x_train,y_train,x_valid,y_valid,k,norm):
    correct=0
    for i in range(0, len(x_valid)):
        pred=KNN_Classifier(x_train,y_train,x_valid[i],k,norm)        
        if(list(pred)==list(y_valid[i])):
            correct=correct+1
    return correct/len(y_valid)

#Function to determine best K-value for a dataset given a norm. Running this function with different
#norms allows us to determine the best norm to use as well. Results are based on validation accuracy.
#Returns the best K value and the maximum validation accuracy.
def BestK_Classification(x_train,y_train,x_valid,y_valid,norm):
    accuracy=[]
    k_values=[]
    for k in range (1,40):
        accuracy=accuracy+[GetValAcc(x_train,y_train,x_valid,y_valid,k,norm)]
        k_values=k_values+[k]
        print(k,accuracy[k-1])
    return (k_values[accuracy.index(max(accuracy))],max(accuracy))

#Returns the test accuracy of KNN on the test set given a training set,K-value and norm.
def GetTestAcc(x_train,y_train,x_test,y_test,k,norm):
    correct=0
    for i in range(0, len(x_test)):
        pred=KNN_Classifier(x_train,y_train,x_test[i],k,norm)        
        if(list(pred)==list(y_test[i])):
            correct=correct+1
    return correct/len(y_test)

#Driver Code

#Finding the best K-Value for 2-norm
print(BestK_Classification(list(xtrain),list(ytrain),xvalid,yvalid,2))
#Test Accuracy for best K-Value
print(GetTestAcc(list(xtrain),list(ytrain),xvalid,yvalid,9,2))

###### Implementation and Performance of KD-Trees (Question 3) #######

#Imports
from sklearn.neighbors import KDTree
import math
import time
from data_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt

#Load data
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('mauna_loa')

#Implementation of KNN for regression using the KDTree data structure from SciKitlearn
def KNN_Tree(x_train,y_train,point):
    rval=0
    tree=KDTree(x_train,metric='euclidean')
    nn=tree.query(point.reshape(1,-1),k=5,return_distance=False)
    for i in nn[0]:
        rval=rval+y_train[i]
    return rval/5

#RMSE Error for KDTree implementation
def Test_Error_Tree(x_train,y_train,x_test,y_test):
    results=[]
    for i in range (0,len(x_test)):
        results=results+[(KNN_Tree(x_train,y_train,x_test[i])-y_test[i])**2]
    return math.sqrt(sum(results)/len(results))

#Function returns the time (in seconds) required to get all test predictions for the Rosenbrock 
#dataset (by the KDTree algorithm) as a function of the dimensionality of the data (dimensionality 
#is an input).
def TimeTaken(d):
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock',n_train=5000,d=d)
    print(np.shape(x_test[0]))
    print(np.shape(x_train[0]))
    time_init=time.time()
    Test_Error_Tree(x_train,y_train,x_test,y_test)
    time_end=time.time()
    return time_end-time_init

#Plots the time taken to get test predictions for different dimensions of the Rosenbrock data.
def PlotTime():
    times=[]
    d=[]
    for i in range(2,60):
        times=times+[TimeTaken(i)]
        d=d+[i]
    plt.plot(np.array(d),np.array(times))
    return 0

#Driver code to generate time plot.
PlotTime()

##### Linear Models for Regression and Classification using SVD (Question 4) ######


#Code for Regression:
#Load data
from data_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('mauna_loa')

#Function to compute weight vector by minimizing least squares loss between the model and actual data.
def SVD_Regression(x_train,y_train):
    #Computes the weight vector by using Economy SVD
    padding=np.ones((np.shape(x_train)[0],1))
    X=np.hstack((padding,x_train))
    u,s,vh=np.linalg.svd(X)
    u1=u[:,0:np.shape(X)[1]]
    w=np.dot(vh.transpose(),np.dot(np.linalg.inv(np.diag(s)),np.dot(u1.transpose(),y_train)))
    return w

#Function to generate a vector of predicted target values for all datapoints in the test set.
def Prediction(x_test,weight):
    padding=np.ones((np.shape(x_test)[0],1))
    X=np.hstack((padding,x_test))
    return np.dot(X,weight)

#Function to compute RMSE for the vector of predicted target values for the test set.
def RMSE(prediction,actual):
    error=np.subtract(actual,prediction)
    rmse=np.sqrt(np.mean(np.square(error),axis=0))
    return rmse

#Driver Code

#Compute weight vector:
weight=SVD_Regression(np.vstack((xtrain,xvalid)),np.vstack((ytrain,yvalid)))
#Predict target values for test set:
p=Prediction(xtest,weight)
#Calculate RMSE between actual and predicted values:
RMSE(p,ytest)

#Code for Classification:
#Load data
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('mnist_small')

#Function to comupte weight vector by minimizing least squares loss between the model and actual data.
#Note that the target values were converted from booleans to integers (0->False and 1->True).
def SVD_Classification(x_train,y_train):
    padding=np.ones((np.shape(x_train)[0],1))
    X=np.hstack((padding,x_train))
    u,s,vh=np.linalg.svd(X)
    u1=u[:,0:np.shape(X)[1]]
    Y=np.multiply(y_train,1)
    Y=np.multiply(Y,2)
    Y=Y-1
    w=np.dot(vh.transpose(),np.dot(np.linalg.inv(np.diag(s)),np.dot(u1.transpose(),Y)))
    return w

#Predict targets for the test set. Return vector of predicted targets. Note that a vector of booleans
#was returned to match the acutal data.
def Prediction_Classification(x_test,w):
    padding=np.ones((np.shape(x_test)[0],1))
    X=np.hstack((padding,x_test))
    pred=np.dot(X,weight)
    for i in pred:
        idx=np.argmax(i)
        i[idx]=True
        i[0:idx]=False
        i[idx+1:len(i)]=False     
    return pred

#Determine accuracy for the predicted values on test set.
def Classification_Acc(prediction,actual):
    acc=0
    for i in range(0,np.shape(prediction)[0]):
        if np.array_equal(prediction[i],actual[i]):
            acc=acc+1
    return acc/len(prediction)

#Driver Code:
weight=SVD_Classification(np.vstack((xtrain,xvalid)),np.vstack((ytrain,yvalid)))
#Predict target values for test set:
p=Prediction_Classification(xtest,weight)
#Calculate accuracy of predicted values:
Classification_Acc(p,ytest)
