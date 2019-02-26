import cv2, numpy as np

webcam = cv2.VideoCapture(0)
i = 1

while True:
    s, imagem = webcam.read()
    imagem = cv2.flip(imagem, 180)
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.05)

    dst = cv2.dilate(dst, None)

    imagem[dst > 0.00001*dst.max()] = [0, 0, 255]
    #gaussianBlur = cv2.GaussianBlur(imagem, (5, 5), 0)
    bilateralFilter = cv2.bilateralFilter(imagem, 9, 75, 75)

    cv2.imshow('Deteccao de Pontos', bilateralFilter)

    cv2.imwrite('sift_keypoints' + str(i) + '.jpg', imagem)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()