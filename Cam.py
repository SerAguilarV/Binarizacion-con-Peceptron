import cv2
from PIL import *
from matplotlib import *
import matplotlib.pyplot as plt
import scipy
from numpy import *
import time

plt.close('all')

def show_webcam():
    cam = cv2.VideoCapture(0)
    #cv2.namedWindow("WebCam",1)
    while True:
        ret_val, img1 = cam.read()
        img = cv2.flip(img1, 1)
        cv2.imshow('WebCam', img)
        #if cv2.waitKey(1) == 27: 
            #break  # esc to quit
        if cv2.waitKey(1) == 32: 
            break
        
    cv2.destroyWindow('WebCam')
    return img1

Img=show_webcam()
#img_to_yuv = cv2.cvtColor(Img,cv2.COLOR_BGR2YUV)
#img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0]) 
#Img = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

img_to_yuv = cv2.cvtColor(Img,cv2.COLOR_BGR2YUV)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_to_yuv[:,:,0]= clahe.apply(img_to_yuv[:,:,0])
Img = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

[fil,col,capa]=Img.shape
Img2=zeros([fil,col,capa])

Img2[:,:,0]=Img[:,:,2]
Img2[:,:,1]=Img[:,:,1]
Img2[:,:,2]=Img[:,:,0]

Img2=uint8(Img2)

plt.figure()
plt.imshow(Img2)
plt.axis('off') 

pos_R=int32(plt.ginput(10)) #Seleccionar 10 puntos de la imagen.
                            #En pares los otros colores
                            #Y nones el o,los que se quieren aislar.
val_R=Img2[pos_R[:,1],pos_R[:,0]]

print(val_R)
plt.close()

# Entrenamiento de neurona

w1=1
w2=-1
w3=1
b1=-1
alpha=0.00000001


for epocs in range(1000):
    for i in range(10):
        a=(w1*val_R[i,0] + w2*val_R[i,1] + w3*val_R[i,2])+b1
       
        if (i%2)==0:
            tar=0
        else:
            tar=1
            
        error=tar-a
             
        w1=w1 + (2*alpha*(error*val_R[i,0]))
        w2=w2 + (2*alpha*(error*val_R[i,1]))
        w3=w3 + (2*alpha*(error*val_R[i,2]))
        b1=b1 + (2*alpha*error)
        
print(w1,w2,w3)
img3=zeros([fil,col])

for i in range(fil):
    for j in range(col):
        a=(w1*double(Img2[i,j,0]) + w2*double(Img2[i,j,1]) + w3*double(Img2[i,j,2]))+b1
        if a<0:
            img3[i,j]=0
        else:
            img3[i,j]=255
            
plt.figure()
plt.imshow(uint8(img3))
plt.gray()
plt.axis('off')
plt.show()
            







