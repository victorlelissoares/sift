import cv2
import numpy as np

webcam = cv2.VideoCapture(0)  # instancia o uso da webcam
sift=cv2.xfeatures2d.SIFT_create()
#surf=cv2.xfeatures2d.SURF_create(1000)

while True:
    s, imagem = webcam.read()  
    imagem = cv2.flip(imagem, 180)  
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) 
    kp=sift.detect(gray, None)
    img=cv2.drawKeypoints(gray, kp, imagem, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp, des = sift.detectAndCompute(gray,None)

    cv2.imshow('Video em tempo real', imagem)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()  
cv2.destroyAllWindows() 
