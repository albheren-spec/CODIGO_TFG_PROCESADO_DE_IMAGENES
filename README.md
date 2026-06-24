
Código fuente del Trabajo Fin de Grado ***Procesado de Imágenes mediante IA:
Clasificación, Autoencoders y Segmentación en Imagen Médica***.

Autor: Alberto Herencia Arce
Grado en Ingeniería Matemática — Universidad Complutense de Madrid
Curso 2025/26

## Descripción

Este repositorio recoge el código de los experimentos de la memoria, que combina el
estudio teórico de las redes neuronales profundas con su aplicación práctica al
procesamiento de imágenes: redes convolucionales (CNN), redes de Kolmogorov-Arnold
(KAN), autoencoders convolucionales y la arquitectura U-Net, aplicadas a clasificación,
reconstrucción y segmentación.

Todos los experimentos están implementados en **PyTorch** y han sido ejecutados en el
entorno GPU T4 de Google Colab.

## Estructura
- **`MNIST/`** : Clasificación de dígitos con CNN y CNN-KAN, y autoencoder convolucional
  (CAE) para reconstrucción.
- **`Cards/`** : Clasificación del palo de cartas de baraja francesa con CNN y CNN-KAN
  (*Cards Image Dataset*).
- **`BRISC/`** : Experimentos sobre el dataset *BRISC 2025* de resonancia magnética
  cerebral: clasificación de tumores con redes preentrenadas (DenseNet121 y AlexNet),
  detección de anomalías mediante autoencoder y segmentación de tumores con U-Net.

## Datasets

- MNIST: disponible a través de `torchvision`.
- Cards Image Dataset: [Kaggle](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification).
- BRISC 2025: [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025).

