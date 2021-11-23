import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA 
from sklearn.decomposition import NMF 

array = np.zeros(shape=[16,20000])
def NOISE(val1,val2,img):
    mean = 0
    var = 0.1
    for i in range(val1,val2):
        noise = np.random.normal(mean, var,(img.shape))
        new=(img+noise).ravel()
        array[:,i] = new
            
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

#Generating Noisy Images
noise_imgA =NOISE(0,1000,imgA)
noise_imgC =NOISE(1000,2000,imgC)
noise_imgD =NOISE(2000,3000,imgD)
noise_imgF =NOISE(3000,4000,imgF) 
noise_imgG =NOISE(4000,5000,imgG)
noise_imgH =NOISE(5000,6000,imgH)
noise_imgI =NOISE(6000,7000,imgI)
noise_imgJ =NOISE(7000,8000,imgJ) 
noise_imgK =NOISE(8000,9000,imgK)
noise_imgL =NOISE(9000,10000,imgL)
noise_imgM =NOISE(10000,11000,imgM)
noise_imgN =NOISE(11000,12000,imgN) 
noise_imgO =NOISE(12000,13000,imgO)
noise_imgP =NOISE(13000,14000,imgP)
noise_imgQ =NOISE(14000,15000,imgQ)
noise_imgR =NOISE(15000,16000,imgR) 
noise_imgS =NOISE(16000,17000,imgS)
noise_imgT =NOISE(17000,18000,imgT)
noise_imgU =NOISE(18000,19000,imgU)
noise_imgV =NOISE(19000,20000,imgV) 

#---------------------------------------------------------------------------

#PCA
pca1=PCA(n_components=16)
pca1.fit_transform(array)
pca1_comp = pca1.components_
pca1_trans = pca1.fit_transform(array)
pca1_vals = pca1.inverse_transform(pca1_trans)
recon1 = pca1_vals.transpose()

plt.figure()
plt.suptitle('Reconstruction using PCA, n_components = 16')
count=0
for k in range(1,21):
    img_k = recon1[count].reshape(4,4)
    plt.subplot(4,5,int(k))
    plt.imshow(img_k)
    count+=1000
    
pca2=PCA(n_components=4)
pca2.fit_transform(array)
pca2_comp = pca2.components_
pca2_trans = pca2.fit_transform(array)
pca2_vals = pca2.inverse_transform(pca2_trans)
recon2 = pca2_vals.transpose()

plt.figure()
plt.suptitle('Reconstruction using PCA, n_components = 4')
count=0
for k in range(1,21):
    img_k = recon2[count].reshape(4,4)
    plt.subplot(4,5,int(k))
    plt.imshow(img_k)
    count+=1000

#----------------------------------------------------------------------------

nmf = NMF(n_components=16)
nmf_trans = nmf.fit_transform(abs(array))
nmf_vals = nmf.inverse_transform(nmf_trans)
recon = nmf_vals.transpose()

plt.figure()
plt.suptitle('Reconstruction using NMF, n_components = 16')
count=0
for k in range(1,21):
    img_k = recon[count].reshape(4,4)
    plt.subplot(4,5,int(k))
    plt.imshow(img_k)
    count+=1000

nmf2 = NMF(n_components=4)
nmf2_trans = nmf2.fit_transform(abs(array))
nmf2_vals = nmf2.inverse_transform(nmf2_trans)
reconn = nmf2_vals.transpose()

plt.figure()
plt.suptitle('Reconstruction using NMF, n_components = 4')
count=0
for k in range(1,21):
    img_k = reconn[count].reshape(4,4)
    plt.subplot(4,5,int(k))
    plt.imshow(img_k)
    count+=1000





