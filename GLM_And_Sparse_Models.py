#Question 2

import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_dataset
#Loading dataset
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('mauna_loa')

def FeatureSpace(x):
    #Construct the feature space matrix for the Mauna-Loa dataset using a set of predetermined basis functions.
    #The feature space consists of linear, exponential and sinusoidal functions.
    features=np.zeros((len(x),70))
    for i in range(0, len(x)):
        for j in range(0,len(features[0])):
            features[i][j]=np.sin((50+j)*x[i])
    return np.hstack((np.ones((len(x),1)),x,np.exp(x),features))


def GLM_Weights(phi_train,y_train,regularizer):
    #Use SVD to compute weights for feature space over training points. The regularizer is used to constrain the maginitude of the weights.
    u,s,vh=np.linalg.svd(phi_train)
    reg=np.identity(len(s))
    reg=regularizer*reg
    u1=u[:,0:np.shape(phi_train)[1]]
    inv=np.linalg.inv(np.dot(np.diag(s).transpose(),np.diag(s))+reg)
    right=np.dot(np.diag(s).transpose(),np.dot(u1.transpose(),y_train))
    w=np.dot(vh.transpose(),np.dot(inv,right))
    return w

def Prediction(phi_test,weight):
    #Compute prediction given testing dataset and precomputed weights.
    return np.dot(phi_test,weight)

def RMSE(prediction,actual):
    #Compute RMSE
    error=np.subtract(actual,prediction)
    rmse=np.sqrt(np.mean(np.square(error),axis=0))
    return rmse

'''Driver Code: Uses the functions above to produce the results required for Question 1. This code can be modified
to apply the code on different datasets or obtain different results'''

#Search over multiple values of the regularizer to determine the best one.
rms=[]
for i in range (1,100):
    pred=Prediction(FeatureSpace(xvalid),GLM_Weights(FeatureSpace(xtrain),ytrain,i))
    rms=rms+[RMSE(pred,yvalid)]
print(min(rms),rms.index(min(rms)))

#Compute prediction over testing dataset using the best regularizer. Plot prediction and compute RMSE.
pred=Prediction(FeatureSpace(xtest),GLM_Weights(FeatureSpace(np.vstack((xvalid,xtrain))),np.vstack((yvalid,ytrain)),11))
print(RMSE(pred,ytest))
plt.plot(xtest,ytest,label="Actual")
plt.plot(xtest,pred,label="Predicted")
plt.title("Actual Data vs Predictions for Generalized Linear Model")
plt.xlabel('Inputs')
plt.ylabel('Outputs')
plt.legend()

#Question 3
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_dataset
#Loading dataset
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('mauna_loa')

def Kernelized_GLM(x_train,y_train,regularizer):
    #Uses Cholesky Decomposition to compute the weights of the kernelized version of the 
    #model constructed in Question 1. First creates the Gram Matrix and then computes weights.
    K=np.dot(FeatureSpace(x_train),FeatureSpace(x_train).transpose())
    reg=np.identity(len(K))
    reg=regularizer*reg
    updated=K+reg
    updated=np.linalg.cholesky(updated)
    a=np.dot(np.linalg.inv(updated),y_train)
    a=np.dot(np.linalg.inv(updated.transpose()),a)
    return a

def Kernelized_Prediction(x_train,x_test,a):
    #Make prediction on the test set using precomputed weights and the Gram Matrix of the training dataset.
    phi_test=FeatureSpace(x_test)
    phi_train=FeatureSpace(x_train)
    return np.dot(phi_test,np.dot(phi_train.transpose(),a))

def Plot_Kernel(x,y):
    #Plot the kernel as a function of one of the inputs to check translational invariance
    phi_x=np.zeros((1,70))
    phi_y=np.zeros((1,70))
    k=[]
    for num in range(0,len(y)):
        for j in range(0,len(phi_x[0])):
            phi_x[0][j]=np.sin((50+j)*x)
            phi_y[0][j]=np.sin((50+j)*y[num])
        X=phi_x.tolist()
        Y=phi_y.tolist()
        X[0]=[1]+[x]+[np.exp(x)]+X[0]
        Y[0]=[1]+[y[num]]+[np.exp(y[num])]+Y[0]
        k=k+[np.dot(np.array(X[0]).transpose(),np.array(Y[0]))]
    plt.plot(y,np.array(k))
    plt.xlabel("z")
    plt.ylabel("k(1,z)")
    
'''Driver code to use the functions above. Can be modified.

pred=Kernelized_Prediction(np.vstack((xvalid,xtrain)),xtest,Kernelized_GLM(np.vstack((xvalid,xtrain)),np.vstack((yvalid,ytrain)),11))
plt.plot(xtest,ytest)
plt.plot(xtest,pred)
z=np.linspace(-0.1,0.1,num=500)
Plot_Kernel(1,z+1)'''

#Question 4
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_dataset
#Loading dataset
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('mauna_loa')

def Gaussian_RBF(x_train,y_train,theta,regularizer):
    #Creates Gram Matrix for a dataset corresponding to a Gaussian Kernel.
    #Computes kernel based on the shape parameter: theta. Inverts Gram Matrix using
    #Cholesky Decomposition to compute weights. The weight sizes are constrained by regularizer.
    
    K=np.zeros((len(x_train),len(x_train)))
    for i in range(0,len(x_train)):
        for j in range(0,len(x_train)):
            K[i][j]=np.sum(np.square(x_train[i]-x_train[j])) 
    K=np.exp(-1*K/theta)

    reg=np.identity(len(K))
    reg=regularizer*reg
    updated=K+reg
    updated=np.linalg.cholesky(updated)
    a=np.dot(np.linalg.inv(updated),y_train)
    a=np.dot(np.linalg.inv(updated.transpose()),a)
    return a

def Gaussian_RBF_Prediction(x_train,x_test,theta,a):
    #Make prediction on the test set using precomputed weights and the Gram Matrix of the training dataset.
    K=np.zeros((len(x_test),len(x_train)))
    for i in range(0,len(x_test)):
        for j in range(0,len(x_train)):
            K[i][j]=np.sum(np.square(np.subtract(x_test[i],x_train[j])))         
    test_K=np.exp(-1*K/theta)
    pred=np.dot(test_K,a)
    return pred

def RMSE(prediction,actual):
    #Compute RMSE
    error=np.subtract(actual,prediction)
    rmse=np.sqrt(np.mean(np.square(error),axis=0))
    return rmse

'''Driver Code that uses above functions. Can be modified'''
#Perform grid search over shape parameters and regularization parameters to determine the best ones based on RMSE
#over validation set.

for t in [0.05,0.1,0.5,1,2]:
    for l in [0.001, 0.01, 0.1, 1]:
        a=Gaussian_RBF(xtrain,ytrain,t,l)
        pred=Gaussian_RBF_Prediction(xtrain,xvalid,t,a)
        print(RMSE(pred,yvalid),t,l)

#Predict the entire Mauna Loa dataset by using the training and validation data.
'''a=Gaussian_RBF(np.vstack((xvalid,xtrain)),np.vstack((yvalid,ytrain)),1,0.001)
pred=Gaussian_RBF_Prediction(np.vstack((xvalid,xtrain)),np.vstack((xvalid,xtrain,xtest)),1,a)
plt.plot(np.vstack((xvalid,xtrain,xtest)),np.vstack((yvalid,ytrain,ytest)),label='Actual')
plt.plot(np.vstack((xvalid,xtrain,xtest)),pred,label='Predicted')
plt.xlabel('Inputs')
plt.ylabel('Outputs')
plt.legend()
plt.title("RBF Regression on Mauna Loa")'''
#Predict the test data using validation and training data. Compute test RMSE.
a=Gaussian_RBF(np.vstack((xvalid,xtrain)),np.vstack((yvalid,ytrain)),1,0.001)
pred=Gaussian_RBF_Prediction(np.vstack((xvalid,xtrain)),xtest,1,a)
print(RMSE(pred,ytest))

#Use the Gaussian RBF network to predict over the Rosenbrock Dataset.
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('rosenbrock',n_train=1000,d=2)

#Grid Search for best parameters based on validation RMSE.
for t in [0.05,0.1,0.5,1,2]:
    for l in [0.001, 0.01, 0.1, 1]:
        a=Gaussian_RBF(xtrain,ytrain,t,l)
        pred=Gaussian_RBF_Prediction(xtrain,xvalid,t,a)
        print(RMSE(pred,yvalid),t,l)
#Make predictions over Rosenbrock test set using best parameters and compute test RMSE.
a=Gaussian_RBF(np.vstack((xvalid,xtrain)),np.vstack((yvalid,ytrain)),2,0.001)
pred=Gaussian_RBF_Prediction(np.vstack((xvalid,xtrain)),xtest,2,a)
RMSE(pred,ytest)

#Use Gaussian RBF model for prediction over a classification dataset.
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('iris')


def Classification_Acc(prediction,actual):
    #Compute classification accuracy given prediction vector and acutal vector.
    acc=0
    for i in range(0,np.shape(prediction)[0]):
        if np.array_equal(prediction[i],actual[i]):
            acc=acc+1
    return acc/len(prediction)

def Classification_Gaussian_RBF_Prediction(x_train,x_test,theta,a):
    #Use Gaussian RBF model to predict outputs for test dataset. The result of simply computing
    #the dot product between the Gram Matrix for the test data and the weights is numerical vector. 
    #The largest entry in this vector is set to TRUE, while all other entries are set to FALSE. This 
    #corresponds to One-Hot-Encoding.
    K=np.zeros((len(x_test),len(x_train)))
    for i in range(0,len(x_test)):
        for j in range(0,len(x_train)):
            K[i][j]=np.sum(np.square(np.subtract(x_test[i],x_train[j])))         
    test_K=np.exp(-1*K/theta)
    pred=np.dot(test_K,a)
    for i in pred:
        idx=np.argmax(i)
        i[idx]=True
        i[0:idx]=False
        i[idx+1:len(i)]=False
    return pred

'''Driver Code for Classification Gaussian RBF.'''
#Grid Search for best parameters.
for t in [0.05,0.1,0.5,1,2]:
    for l in [0.001, 0.01, 0.1, 1]:
        a=Gaussian_RBF(xtrain,ytrain,t,l)
        pred=Classification_Gaussian_RBF_Prediction(xtrain,xvalid,t,a)
        print(Classification_Acc(pred,yvalid),t,l)
#Test predictions using best paramaters.
a=Gaussian_RBF(np.vstack((xvalid,xtrain)),np.vstack((yvalid,ytrain)),2,1)
pred=Classification_Gaussian_RBF_Prediction(np.vstack((xvalid,xtrain)),xtest,2,a)
Classification_Acc(pred,ytest)

#Question 5: Implementation of Orthogonal Matching Pursuit to produce a sparse model consisting of Gaussians Centered
#at chosen training points. Algorithm is used for predicting over the Rosenbrock dataset.

import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_dataset
#Load Dataset
xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('rosenbrock',n_train=200,d=2)

#Orthogonal Matching Pursuit Greddy Algorithm: 
    #Creates a dictiionary of Gaussians with shapes theta centered at each training point.
    #A Gaussian is then picked from the dictionary (based on a heuristic).
    #The dictionary and set of selected functions is updated.
    #The MDL Criteria is computed at each iteration and is plotted at the very end.
    #The number of iterations is passed in with the end variable.
    #Run initially with a large value for end (approximately the size of the dataset) to
    #determine where the minimum MDL is.
    #Once minimum MDL is found set the end value to the corresponding model complexity of the minimum MDL value
    #The function returns the selected functions and computed weights for the sparse model.
def OMP(x_train,y_train,theta,end):
    residual=-1*y_train
    k=0
    candidates=OMPFeatureSpace(x_train,x_train,theta)
    selected=np.array([])
    weights=np.array([])
    iterations=[]
    criteria=[]
    for i in range(0,end):
        k=k+1
        candidates,selected,weights=Heuristic(x_train,y_train,candidates,selected,residual,weights,theta)
        residual=y_train-np.dot(OMPFeatureSpace(x_train,selected,theta),weights)
        criteria.append(0.5*len(x_train)*np.log(np.sum(np.square(residual)))+0.5*k*np.log(len(x_train)))
        iterations.append(k)
    plt.plot(np.array(iterations),np.array(criteria))
    plt.xlabel("Iterations")
    plt.ylabel("MDL Criteria Value")
    plt.title("Model Complexity vs Predicted Generalization Error")
    return selected,weights

#Heuristic for OMP. Returns new set of candidates, selected functions and newly computed weights. Weights are computed
#using QR factorization.
def Heuristic(x_train,y_train,candidates,selected,residual,weights,theta):
    prev=0
    for i in range(0,len(candidates[0])):
        update=(np.dot(residual.transpose(),candidates[:,i].transpose())**2)/(np.dot(candidates[:,i],candidates[:,i].transpose()))
        if(update[0]>prev):
            prev=update[0]
            idx=i
    if(len(weights)==0):
        selected=np.array(np.array(x_train[np.where(candidates[:,idx]==1)]))
    else:
        selected=np.vstack((selected,x_train[np.where(candidates[:,idx]==1)]))
    phi=OMPFeatureSpace(x_train,selected,theta)
    q,r=np.linalg.qr(phi)
    weights=np.dot(np.linalg.inv(r),np.dot(q.transpose(),y_train))
    #print(weights)
    candidates=np.delete(candidates,idx,1)
    return candidates,selected,weights
#Helper functions for OMP and Heuristic.
def MakeFeatureSpace(x_train,theta):
    K=np.zeros((len(x_train),len(x_train)))
    for i in range(0,len(x_train)):
            K[i]=np.exp(-1*(np.linalg.norm(x_train-x_train[i],axis=1)**2)/theta)    
    return K

def OMPFeatureSpace(x_train,selected,theta):
    K=np.zeros((len(x_train),len(selected)))
    for i in range(0,len(x_train)):
            K[i]=np.exp(-1*(np.linalg.norm(selected-x_train[i],axis=1)**2)/theta)
    return K

#Make test prediction based on selected basis functions and computed weights.
def OMP_Prediction(selected,x_test,theta,weights):
    K=np.zeros((len(x_test),len(selected)))
    for i in range(0,len(x_test)):
        for j in range(0,len(selected)):
            K[i][j]=np.sum(np.square(np.subtract(x_test[i],selected[j])))         
    test_K=np.exp(-1*K/theta)
    pred=np.dot(test_K,weights)
    return pred
#Compute RMSE.
def RMSE(prediction,actual):
    error=np.subtract(actual,prediction)
    rmse=np.sqrt(np.mean(np.square(error),axis=0))
    return rmse

'''Driver Code for OMP. Can be changed to test models with different shape parameters or complexities. The one below is for
a shape parameter of 0.01 and 800 iterations.'''
#Construct sparse model.
selected,weights=OMP(np.vstack((xtrain,xvalid)),np.vstack((ytrain,yvalid)),0.01,800)
#Make test predictions based on sparse model.
pred=OMP_Prediction(selected,xtest,0.01,weights)
#Compute RMSE.
print(RMSE(ytest,pred))
