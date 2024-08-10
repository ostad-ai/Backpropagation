#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import pickle
from math import exp,tanh
#---------------------
# Backpropagation with pure python, written by Hamed Shah-Hosseini
# This is a three-layer MLP with bias for both input and hidden layers
# version 0.0
# if you use this in a research or application, you should reference the source
#-------------guidance:
# hidden layer can have: sigmoid, relu, and tanh as activation functions
# output layer can have: sigmoid, relu, tanh, and linear (identity) as activation functions
class BP:
    def __init__(self,etta=.01,Ninput=4, Nhidden=12, Noutput=3,
                 act_hidden='tanh',act_output='sigmoid'):
        self.etta=etta #learning rate
        self.act_hidden=act_hidden; self.act_output=act_output
        self.Ninput,self.Nhidden,self.Noutput=Ninput+1,Nhidden+1,Noutput
        self.wh=[[.1*random.random() for i in range(self.Ninput)] for j in range(self.Nhidden)]
        self.wo=[[.1*random.random() for i in range(self.Nhidden)] for j in range(self.Noutput)]
    def train_online(self,x,d): # both x and d are lists
        if len(x)!=self.Ninput-1:
            print('Wrong size of input vector')
            return
        elif len(d)!=self.Noutput:
            print('Wrong size of desired output vector')
            return
        vh,oh,vout,y=self.forward_train(x)  #current output of net for input x
        e_outs=[]
        for j in range(self.Noutput):
            e_outs.append(self.phi_prime(vout[j],False)*(d[j]-y[j]))
        e_hiddens=[0]
        for j in range(1,self.Nhidden):
            dj=0
            for k in range(self.Noutput):
                dj+=self.wo[k][j]*e_outs[k]
            e_hiddens.append(self.phi_prime(vh[j])*dj)
        #--------------update weights
        for j in range(self.Noutput):
            for  i in range(self.Nhidden):
                self.wo[j][i]+=self.etta*e_outs[j]*oh[i]
        for j in range(1,self.Nhidden):            
            for  i in range(self.Ninput):
                if i==0:
                    self.wh[j][i]+=self.etta*e_hiddens[j]
                else:
                    self.wh[j][i]+=self.etta*e_hiddens[j]*x[i-1]
    def forward_train(self,x): # train phase: compute output for input vector x
        oh=[1]; vh=[1];vout=[]
        for i in range(1,self.Nhidden):
            v=self.wh[i][0]  # for bias term
            for j in range(1,self.Ninput):
                v+=self.wh[i][j]*x[j-1]
            vh.append(v)
            o=self.phi(v,True) #true for hidden activation function
            oh.append(o)
        oo=[]
        for i in range(self.Noutput):
            v=self.wo[i][0]
            for j in range(1,self.Nhidden):
                v+=self.wo[i][j]*oh[j]
            vout.append(v)
            o=self.phi(v,False)
            oo.append(o)
        return vh,oh,vout,oo
    def forward(self,x): # test phase: compute output for input vector x
        oh=[1]
        for i in range(1,self.Nhidden):
            v=self.wh[i][0]
            for j in range(1,self.Ninput):
                v+=self.wh[i][j]*x[j-1]
            o=self.phi(v,True) #true for hidden activation fucntion
            oh.append(o)
        oo=[]
        for i in range(self.Noutput):
            v=self.wo[i][0]
            for j in range(1,self.Nhidden):
                v+=self.wo[i][j]*oh[j]
            o=self.phi(v,False)
            oo.append(o)
        return oo
    def tanH(self,v):
        return tanh(v)
    def sigmoid(self,v):
        return 1/(1+exp(-v))
    def ReLU(self,x):
        if x>0:
            return x
        else:
            return 0
    def phi_prime(self,v,hidden=True):
        if hidden:
            if self.act_hidden=='sigmoid':
                y=self.sigmoid(v)
                return y*(1-y)
            elif self.act_hidden=='tanh':
                y=self.tanH(v)
                return (1-y)*(1+y)
            elif self.act_hidden=='relu':
                if v>0:
                    return 1
                else:
                    return 0
        else:
            if self.act_output=='sigmoid':
                y=self.sigmoid(v)
                return y*(1-y)
            elif self.act_output=='tanh':
                y=self.tanH(v)
                return (1-y)*(1+y)
            elif self.act_output=='relu':
                if v>0:
                    return 1
                else:
                    return 0
            elif self.act_output=='linear':
                return 1
            
    def phi(self,v,hidden=True):
        if hidden:
            if self.act_hidden=='sigmoid':
                return self.sigmoid(v)
            elif self.act_hidden=='tanh':
                return self.tanH(v)
            elif self.act_hidden=='relu':
                return self.ReLU(v)
        else:
            if self.act_output=='sigmoid':
                return self.sigmoid(v)
            elif self.act_output=='tanh':
                return self.tanH(v)
            elif self.act_output=='relu':
                return self.ReLU(v)
            elif self.act_output=='linear':
                return v

def save_model(bp_model=None,file_path='./mymodel.pkl'):
        if bp_model:
            with open(file_path,'wb') as file:
                pickle.dump(bp_model,file)
            
def load_model(file_path='./mymodel.pkl'):
    with open(file_path,'rb') as file:
        return pickle.load(file)

