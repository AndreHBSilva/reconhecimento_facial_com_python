import cv2
import os
import numpy as np

#cria os reconhecedores
eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemPeloNome():
  caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
  print(caminhos)
  faces = []
  nomes = []
  
  for caminhoImagem in caminhos:
    imagemFace = cv2.imread(caminhoImagem)
    imagemFaceCinza = cv2.cvtColor(imagemFace, cv2.COLOR_BGR2GRAY)
    nome = os.path.split(caminhoImagem)[-1].split('_')[0] #quebra onde tem _
    print(nome)
    if 'Andre' in nome:
      nomes.append(1)
    else:
      nomes.append(2)
    faces.append(imagemFaceCinza)
  return np.array(nomes), faces

nomes, faces = getImagemPeloNome()
print(nomes)
eigenface.train(faces, nomes)
eigenface.write('classificadoreseigen.yml')
    