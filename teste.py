import cv2

camera = cv2.VideoCapture(0)

while cv2.waitKey(1) == -1:  # enquanto não for pressionado nenhuma tecla
    status, imagem = camera.read()  # lê o estado da câmera e a imagem capturada
    cv2.imshow("Face detectada", imagem)
    print(status)

camera.release()
cv2.destroyAllWindows()
