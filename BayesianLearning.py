#Question 1 Part a

from data_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt

xtrain, xvalid, xtest, ytrain, yvalid, ytest=load_dataset('iris')

#Functions to compute the MAP estimate of the posterior distribution
def sigmoid(w,x_i):
    return 1/(1+np.exp(-1*np.dot(w.transpose(),x_i)))

def RegularizedBCELoss(w,x,y,variance):
    dim=len(x[0])
    rval=-0.5*(dim+1)*(np.log(2*variance*np.pi))-np.sum(np.square(w))/(2*variance)
    for i in range(0,len(x)):
        rval=rval+y[i]*np.log(sigmoid(w,x[i]))+(1-y[i])*np.log(1-sigmoid(w,x[i]))
    return -1*rval

def RegularizedBCEGrad(w,x,y,variance):
    rval=np.zeros((1,len(x[0])))
    for i in range(0,len(x)):
        rval=rval+(y[i]-sigmoid(w,x[i]))*x[i]
    rval=rval.transpose()-w/variance
    return rval

def FullBatchGD(x_train,y_train,lr,init_values,converged,variance):
    rval=init_values
    iters=[]
    losses=[]
    accum=0
    while(np.linalg.norm(RegularizedBCEGrad(rval,x_train,y_train,variance))>converged):
        iters=iters+[accum]
        accum=accum+1
        losses=losses+[RegularizedBCELoss(rval,x_train,y_train,variance)]
        rval=rval+lr*RegularizedBCEGrad(rval,x_train,y_train,variance)  
    plt.plot(iters,losses)
    return (rval,losses,iters)

#Preparing Dataset: Merging validation and training datasets. 
#Converting multiclass targets to binary targets.
xtrain=np.hstack((np.ones((len(xtrain),1)),xtrain))
xvalid=np.hstack((np.ones((len(xvalid),1)),xvalid))
xtest=np.hstack((np.ones((len(xtest),1)),xtest))
xtrain=np.vstack((xtrain,xvalid))
ytrain=1*ytrain[:,(1,)]
yvalid=1*yvalid[:,(1,)]
ytrain=np.vstack((ytrain,yvalid))
ytest=1*ytest[:,(1,)]
init_weights=np.zeros((len(xtrain[0]),1))

#Finding MAP solution for a Posterior corresponding to a Gaussain prior
#with variance 1.
w,loss,iters=FullBatchGD(xtrain,ytrain,0.001,init_weights,0.001,1)
#Computing the Log-Likelihood based on Bernoulli Classification.
def LogLikelihood(w,x,y):
    rval=0
    for i in range(0,len(x)):
        rval=rval+y[i]*np.log(sigmoid(w,x[i]))+(1-y[i])*np.log(1-sigmoid(w,x[i]))
    return rval
#Compute the Hessian for computing the Laplace Approximation.
def hessian(x_train,w,variance):
    h_likelihood=np.zeros((len(x_train[0]),len(x_train[0])))
    h_prior=-1*np.identity(len(x_train[0]))/variance
    for i in x_train:
        h_likelihood=h_likelihood+sigmoid(w,i)*(sigmoid(w,i)-1)*np.outer(i,i.transpose())
    return h_likelihood+h_prior
#Compute the Marginal Likelihood based on the Laplace Approximation.
def marginal(x_train,w,variance,y_train):
    det=np.linalg.det(-1*hessian(x_train,w,variance))
    dim=len(x_train[0])
    rval=LogLikelihood(w,x_train,y_train)-0.5*dim*(np.log(2*variance*np.pi))-np.sum(np.square(w))/(2*variance)
    return rval+0.5*dim*np.log(2*np.pi)-0.5*np.log(det)
    
marginal(xtrain,w,1,ytrain)

#Question 1 part b (Uses scipy.stats for sampling)
from scipy.stats import matrix_normal
from scipy.stats import multivariate_normal
import numpy as np

#Define a gaussian proposal
def Proposal(mean,cov,res):
    rval=matrix_normal.rvs(mean=mean,rowcov=cov)
    for i in range(1,res):
        rval=np.hstack((rval,matrix_normal.rvs(mean=mean,rowcov=cov,random_state=i)))
    return rval

#Compute the posterior
def Posterior(x,y,weight,mean,cov):
    likelihood=1
    for i in range(0,len(x)):
        likelihood=likelihood*(sigmoid(weight,x[i])**(y[i][0]))*((1-sigmoid(weight,x[i]))**(1-y[i][0]))
    return likelihood*multivariate_normal.pdf(weight,mean.transpose()[0],cov)

#Compute a prediction for a single data-point using importance sampling
def Single_Prediction(x,x_train,y_train,mean,cov,res):
    samples=Proposal(mean,cov,res)
    impweight=np.zeros((res,1))
    for i in range(0,res):
        impweight[i]=Posterior(x_train,y_train,samples[:,i],np.zeros((5,1)),np.identity(5))
        impweight[i]=impweight[i]/multivariate_normal.pdf(samples[:,i],mean.transpose()[0],cov)
    total=np.sum(impweight)
    zero_prob=0
    one_prob=0
    for i in range(0,res):
        one_prob+=sigmoid(samples[:,i],x)*impweight[i]/total
        zero_prob+=(1-sigmoid(samples[:,i],x))*impweight[i]/total
    if(zero_prob>one_prob):
        return 0
    return 1

#Compute predictions for multiple data-points.
def TestPrediction(x_test,x_train,y_train,mean,cov,res):
    rval=np.zeros((len(x_test),1))
    for i in range(0,len(x_test)):
        rval[i]=Single_Prediction(x_test[i],x_train,y_train,mean,cov,res)
    return rval

#Determine the accuracy of the model when run on any dataset.
def Classification_Acc(prediction,actual):
    acc=0
    for i in range(0,np.shape(prediction)[0]):
        if np.array_equal(prediction[i],actual[i]):
            acc=acc+1
    return acc/len(prediction)

#Computing accuracy on the training set.
Classification_Acc(TestPrediction(xtrain,xtrain,ytrain,w,-10*np.linalg.inv(hessian(xtrain,w,1)),100),ytrain)

