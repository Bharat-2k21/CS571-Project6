import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import decomposition

def NOISE(N):
    mean = 0
    var = 0.1
    noise = np.random.normal(mean, var,(N.shape))
    return N + noise
    
#Generating Images
plt.suptitle('20 Original 4x4 Images')
plt.subplot(451)
imgA=np.array([[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1]])
plt.imshow(imgA)
plt.subplot(452)
imgC=np.array([[0,1,1,0],[1,0,0,0],[1,0,0,0],[0,1,1,0]])
plt.imshow(imgC)
plt.subplot(453)
imgD=np.array([[1,1,0,0],[1,0,1,0],[1,0,1,0],[1,1,0,0]])
plt.imshow(imgD)
plt.subplot(454)
imgF=np.array([[1,1,1,0],[1,0,0,0],[1,1,1,0],[1,0,0,0]])
plt.imshow(imgF)
plt.subplot(455)
imgG=np.array([[0,1,1,0],[1,0,0,0],[1,0,1,1],[0,1,1,0]])
plt.imshow(imgG)

plt.subplot(456)
imgH=np.array([[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1]])
plt.imshow(imgH)
plt.subplot(457)
imgI=np.array([[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]])
plt.imshow(imgI)
plt.subplot(458)
imgJ=np.array([[1,1,1,0],[0,0,1,0],[1,0,1,0],[0,1,1,0]])
plt.imshow(imgJ)
plt.subplot(459)
imgK=np.array([[1,0,0,0],[1,0,1,0],[1,1,0,0],[1,0,1,0]])
plt.imshow(imgK)
plt.subplot(4,5,10)
imgL=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,0]])
plt.imshow(imgL)

plt.subplot(4,5,11)
imgM=np.array([[0,0,0,0],[1,0,1,0],[0,1,0,0],[1,0,1,0]])
plt.imshow(imgM)
plt.subplot(4,5,12)
imgN=np.array([[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1]])
plt.imshow(imgN)
plt.subplot(4,5,13)
imgO=np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]])
plt.imshow(imgO)
plt.subplot(4,5,14)
imgP=np.array([[1,1,1,0],[1,0,1,0],[1,1,1,0],[1,0,0,0]])
plt.imshow(imgP)
plt.subplot(4,5,15)
imgQ=np.array([[1,1,1,0],[1,0,1,0],[1,1,1,0],[0,0,0,1]])
plt.imshow(imgQ)

plt.subplot(4,5,16)
imgR=np.array([[1,1,1,0],[1,0,1,0],[1,1,0,0],[1,0,1,0]])
plt.imshow(imgR)
plt.subplot(4,5,17)
imgS=np.array([[1,1,1,0],[1,0,0,0],[0,0,1,0],[1,1,1,0]])
plt.imshow(imgS)
plt.subplot(4,5,18)
imgT=np.array([[1,1,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]])
plt.imshow(imgT)
plt.subplot(4,5,19)
imgU=np.array([[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]])
plt.imshow(imgU)
plt.subplot(4,5,20)
imgV=np.array([[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]])
plt.imshow(imgV)
#----------------------------------------------------------------------------

#Generating Noisy Images
noise_imgA =NOISE(imgA)
noise_imgC =NOISE(imgC)
noise_imgD =NOISE(imgD)
noise_imgF =NOISE(imgF) 
noise_imgG =NOISE(imgG)
noise_imgH =NOISE(imgH)
noise_imgI =NOISE(imgI)
noise_imgJ =NOISE(imgJ) 
noise_imgK =NOISE(imgK)
noise_imgL =NOISE(imgL)
noise_imgM =NOISE(imgM)
noise_imgN =NOISE(imgN) 
noise_imgO =NOISE(imgO)
noise_imgP =NOISE(imgP)
noise_imgQ =NOISE(imgQ)
noise_imgR =NOISE(imgR) 
noise_imgS =NOISE(imgS)
noise_imgT =NOISE(imgT)
noise_imgU =NOISE(imgU)
noise_imgV =NOISE(imgV) 

plt.figure()
plt.suptitle('Noisy Images')
plt.subplot(451)
plt.imshow(noise_imgA)
plt.subplot(452)
plt.imshow(noise_imgC)
plt.subplot(453)
plt.imshow(noise_imgD)
plt.subplot(454)
plt.imshow(noise_imgF)
plt.subplot(455)
plt.imshow(noise_imgG)
plt.subplot(456)
plt.imshow(noise_imgH)
plt.subplot(457)
plt.imshow(noise_imgI)
plt.subplot(458)
plt.imshow(noise_imgJ)
plt.subplot(459)
plt.imshow(noise_imgK)
plt.subplot(4,5,10)
plt.imshow(noise_imgL)

plt.subplot(4,5,11)
plt.imshow(noise_imgM)
plt.subplot(4,5,12)
plt.imshow(noise_imgN)
plt.subplot(4,5,13)
plt.imshow(noise_imgO)
plt.subplot(4,5,14)
plt.imshow(noise_imgP)
plt.subplot(4,5,15)
plt.imshow(noise_imgQ)
plt.subplot(4,5,16)
plt.imshow(noise_imgR)
plt.subplot(4,5,17)
plt.imshow(noise_imgS)
plt.subplot(4,5,18)
plt.imshow(noise_imgT)
plt.subplot(4,5,19)
plt.imshow(noise_imgU)
plt.subplot(4,5,20)
plt.imshow(noise_imgV)
#----------------------------------------------------------------------------

#Generating and appending Multiple Noisy Images and Performing PCA
n_components = 3
estimator = decomposition.PCA(n_components=n_components, svd_solver='randomized')
estimator2 = decomposition.NMF(n_components=n_components, init='random')

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgA)
    arraynew=np.append(arraynew,array)

newarray_A=arraynew.reshape(1000,16).transpose()
imgA_recons = estimator.inverse_transform(estimator.fit_transform(newarray_A))
A = imgA_recons[:,0].reshape(4,4)
plt.figure()
plt.suptitle('Reconstruction using PCA')
plt.subplot(451)
plt.imshow(A)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgC)
    arraynew=np.append(arraynew,array)

newarray_C=arraynew.reshape(1000,16).transpose()
imgC_recons = estimator.inverse_transform(estimator.fit_transform(newarray_C))
C = imgC_recons[:,0].reshape(4,4)
plt.subplot(452)
plt.imshow(C)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgD)
    arraynew=np.append(arraynew,array)

newarray_D=arraynew.reshape(1000,16).transpose()
imgD_recons = estimator.inverse_transform(estimator.fit_transform(newarray_D))
D = imgD_recons[:,0].reshape(4,4)
plt.subplot(453)
plt.imshow(D)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgF)
    arraynew=np.append(arraynew,array)

newarray_F=arraynew.reshape(1000,16).transpose()
imgF_recons = estimator.inverse_transform(estimator.fit_transform(newarray_F))
F = imgF_recons[:,0].reshape(4,4)
plt.subplot(454)
plt.imshow(F)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgG)
    arraynew=np.append(arraynew,array)

newarray_G=arraynew.reshape(1000,16).transpose()
imgG_recons = estimator.inverse_transform(estimator.fit_transform(newarray_G))
G = imgG_recons[:,0].reshape(4,4)
plt.subplot(455)
plt.imshow(G)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgH)
    arraynew=np.append(arraynew,array)

newarray_H=arraynew.reshape(1000,16).transpose()
imgH_recons = estimator.inverse_transform(estimator.fit_transform(newarray_H))
H = imgH_recons[:,0].reshape(4,4)
plt.subplot(456)
plt.imshow(H)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgI)
    arraynew=np.append(arraynew,array)

newarray_I=arraynew.reshape(1000,16).transpose()
imgI_recons = estimator.inverse_transform(estimator.fit_transform(newarray_I))
I = imgI_recons[:,0].reshape(4,4)
plt.subplot(457)
plt.imshow(I)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgJ)
    arraynew=np.append(arraynew,array)

newarray_J=arraynew.reshape(1000,16).transpose()
imgJ_recons = estimator.inverse_transform(estimator.fit_transform(newarray_J))
J = imgJ_recons[:,0].reshape(4,4)
plt.subplot(458)
plt.imshow(J)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgK)
    arraynew=np.append(arraynew,array)

newarray_K=arraynew.reshape(1000,16).transpose()
imgK_recons = estimator.inverse_transform(estimator.fit_transform(newarray_K))
K = imgK_recons[:,0].reshape(4,4)
plt.subplot(459)
plt.imshow(K)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgL)
    arraynew=np.append(arraynew,array)

newarray_L=arraynew.reshape(1000,16).transpose()
imgL_recons = estimator.inverse_transform(estimator.fit_transform(newarray_L))
L = imgL_recons[:,0].reshape(4,4)
plt.subplot(4,5,10)
plt.imshow(L)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgM)
    arraynew=np.append(arraynew,array)

newarray_M=arraynew.reshape(1000,16).transpose()
imgM_recons = estimator.inverse_transform(estimator.fit_transform(newarray_M))
M = imgM_recons[:,0].reshape(4,4)
plt.subplot(4,5,11)
plt.imshow(M)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgN)
    arraynew=np.append(arraynew,array)

newarray_N=arraynew.reshape(1000,16).transpose()
imgN_recons = estimator.inverse_transform(estimator.fit_transform(newarray_N))
N = imgN_recons[:,0].reshape(4,4)
plt.subplot(4,5,12)
plt.imshow(N)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgO)
    arraynew=np.append(arraynew,array)

newarray_O=arraynew.reshape(1000,16).transpose()
imgO_recons = estimator.inverse_transform(estimator.fit_transform(newarray_O))
O = imgO_recons[:,0].reshape(4,4)
plt.subplot(4,5,13)
plt.imshow(O)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgP)
    arraynew=np.append(arraynew,array)

newarray_P=arraynew.reshape(1000,16).transpose()
imgP_recons = estimator.inverse_transform(estimator.fit_transform(newarray_P))
P = imgP_recons[:,0].reshape(4,4)
plt.subplot(4,5,14)
plt.imshow(P)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgQ)
    arraynew=np.append(arraynew,array)

newarray_Q=arraynew.reshape(1000,16).transpose()
imgQ_recons = estimator.inverse_transform(estimator.fit_transform(newarray_Q))
Q = imgQ_recons[:,0].reshape(4,4)
plt.subplot(4,5,15)
plt.imshow(Q)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgR)
    arraynew=np.append(arraynew,array)

newarray_R=arraynew.reshape(1000,16).transpose()
imgR_recons = estimator.inverse_transform(estimator.fit_transform(newarray_R))
R = imgR_recons[:,0].reshape(4,4)
plt.subplot(4,5,16)
plt.imshow(R)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgS)
    arraynew=np.append(arraynew,array)

newarray_S=arraynew.reshape(1000,16).transpose()
imgS_recons = estimator.inverse_transform(estimator.fit_transform(newarray_S))
S = imgS_recons[:,0].reshape(4,4)
plt.subplot(4,5,17)
plt.imshow(S)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgT)
    arraynew=np.append(arraynew,array)

newarray_T=arraynew.reshape(1000,16).transpose()
imgT_recons = estimator.inverse_transform(estimator.fit_transform(newarray_T))
T = imgT_recons[:,0].reshape(4,4)
plt.subplot(4,5,18)
plt.imshow(T)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgU)
    arraynew=np.append(arraynew,array)

newarray_U=arraynew.reshape(1000,16).transpose()
imgU_recons = estimator.inverse_transform(estimator.fit_transform(newarray_U))
U = imgU_recons[:,0].reshape(4,4)
plt.subplot(4,5,19)
plt.imshow(U)

arraynew=np.array([])
for i in range(1000):
    array=NOISE(imgV)
    arraynew=np.append(arraynew,array)

newarray_V=arraynew.reshape(1000,16).transpose()
imgV_recons = estimator.inverse_transform(estimator.fit_transform(newarray_V))
V = imgV_recons[:,0].reshape(4,4)
plt.subplot(4,5,20)
plt.imshow(V)

