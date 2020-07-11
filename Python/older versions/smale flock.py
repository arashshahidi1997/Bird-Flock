#!/usr/bin/env python
# coding: utf-8

# In[731]:


import numpy as np
import matplotlib.pyplot as plt


# In[749]:


n=100       #number of birds
d=2          #number of dimensions
p0=np.random.rand(n,d)
q0=np.random.rand(n,d)


# In[750]:


#parameters
kappa=1
sigma=.1
beta=0.2


# # Functions

# In[751]:


def gamma(p,q):            #sum of distances(in place and velocities)
    s=0
    r=0
    for i in range(n):
        for j in range(i+1,n):
            for k in range(d):
                s=s+(q[i,k]-q[j,k])**2
                r=r+(p[i,k]-p[j,k])**2
    return [s,r]


# In[752]:


def smale(i,k0,p,q):
    s = 0
    for j in range(n):
        r = 0
        for k in range(d):
            r = r + (q[i, k]-q[j, k])**2
        a = kappa/(np.power((sigma*sigma+r), beta))
        s = s+a*(p[j, k0]-p[i, k0])
    return s


# In[753]:


print(gamma(p0,q0))


# # Iteration

# In[754]:


T=100
gamma_t=np.ndarray((T,2))
q=np.ndarray((n,d))
p=np.ndarray((n,d))
q=q0
p=p0
for t in range(T):
    qprime=np.ndarray((n,d))
    pprime=np.ndarray((n,d))
    for i in range (n):
        for k0 in range(d):
            qprime[i,k0]=q[i,k0]+p[i,k0]
            pprime[i,k0]=p[i,k0]+smale(i,k0,p,q)
   # for i in range(n):
   #     for k in range(d):
   #         pprime[i,k]=pprime[i,k]/np.linalg.norm(pprime[i,:])
    gamma_t[t,0]=gamma(pprime,qprime)[0]
    gamma_t[t,1]=gamma(pprime,qprime)[1]
    p=pprime
    q=qprime


# In[755]:


print(gamma_t[T-1])


# In[756]:


h=np.ndarray((T-10,2))
for i in range(t-10):
    h[i,:]=gamma_t[i+10,:]


# In[757]:


plt.plot(gamma_t[:,0])
plt.ylabel('gamma_t')
plt.show()


# In[758]:


plt.plot(gamma_t[:,1])
plt.ylabel('lambda_t')
plt.show()


# # viscek model

# In[829]:


phi0=2*np.pi*np.random.rand(n)
x0=1000*np.random.rand(n,2)
r=1


# In[830]:


def vis(i,x,phi):
    s=0
    for j in range(n):
        if (x[j,0]-x[i,0])**2+(x[j,1]-x[i,1])**2<r:
            s=s+phi[j]
    return s/n


# In[831]:


def delta(phi,x):            #sum of distances(in place and velocities)
    r=0
    s=0
    for i in range(n):
        for j in range(n):
                r=r+(phi[i]-phi[j])**2
                s=s+(x[j,0]-x[i,0])**2+(x[j,1]-x[i,1])**2
    return [s/2,r/2]


# In[832]:


T=10
delta_t=np.ndarray((T,2))
x=np.ndarray((n,2))
phi=np.zeros(n)
x=x0
phi=phi0
for t in range(T):
    xprime=np.ndarray((n,2))
    phiprime=np.zeros(n)
    for i in range (n):
            xprime[i,0]=x[i,0]+np.cos(phi[i])
            xprime[i,1]=x[i,1]+np.sin(phi[i])
            phiprime[i]=vis(i,x,phi)%(2*np.pi)
    delta_t[t,0]=delta(phiprime,xprime)[0]
    delta_t[t,1]=delta(phiprime,xprime)[1]
    x=xprime
    phi=phiprime


# In[833]:


print(delta_t[T-1])


# In[834]:


plt.plot(delta_t[:,0])
plt.ylabel('delta_t')
plt.show()


# In[835]:


plt.plot(delta_t[:,1])
plt.ylabel('delta_t')
plt.show()


# In[ ]: