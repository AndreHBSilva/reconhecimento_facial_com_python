import cv2
import numpy as np
classificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

amostra = 1 #número de amostra atual
numeroMaxAmostras = 25 #número total de amostras para treinar a IA
nome = input("Digite seu nome: ") #nome que irá aparecer na face reconhecida

altura, largura = 220, 220 #tamanho padrão para a imagem de treinamento

while True:
  status, imagem = camera.read()
  
  imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
  facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor = 1.5, minSize=(150,150))
  for (x, y, altura, largura) in facesDetectadas:
    cv2.rectangle(imagem, (x,y), (x+largura, y+altura), (0,0,255), 2)
    if cv2.waitKey(1) & 0xFF == ord('q'): #se apertou a tecla 'q', então salva a imagem
      """
      Verifica se a média dos valores RGB da imagem são maiores que 110. Ou seja, verifica se a luminosidade
      da imagem está aceitável para usarmos no treinamento. Lembrando que os valores RGB vão de 0 a 255.
      """
      print('Média: ', np.average(imagemCinza))
      if np.average(imagemCinza) > 110:
        imagemFace = cv2.resize(imagemCinza[y: y+altura, x: x+altura], (largura, altura))
        
        #local que ficará cada foto tirada
        localFoto = "fotos/"+str(nome.encode("utf-8"))+"_"+str(amostra)+".jpg"
        print(nome.encode("utf-8"))
        #grava a foto na pasta
        cv2.imwrite(localFoto, imagemFace)
        amostra += 1
        
  cv2.imshow("Face detectada", imagem)
  if amostra > numeroMaxAmostras:
    break
  
print("Fotos capturadas com sucesso.")
camera.release()
cv2.destroyAllWindows()
      
  