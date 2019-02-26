import cv2, csv, os, time, numpy as np, sys, sklearn

#caminho para esse diretório, caso seja diferente, cole o seu entre os ''
path='C:U/sers/v/Dropbox/sift'
os.chdir(path)
i=1

#Função para salvar arquivo com data e hora
'''titulo = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")
saida = open('face_recon_'+titulo+'.csv', 'w')
export = csv.writer(saida, quoting=csv.QUOTE_NONNUMERIC)'''

file_list=[]

for file in os.listdir(path):
    if file.endswith(".jpg"):
        file_list.append(file)

for file in file_list:
	img = cv2.imread(file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray,None)
	img=cv2.drawKeypoints(gray,kp,img)
	cv2.imwrite('sift_keypoints'+str(i)+'.jpg',img)
	kp, des = sift.detectAndCompute(gray,None)
	f=open("point"+str(i)+".txt", "w+")
	f.write(str(kp))
	f.close()
