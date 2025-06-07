#import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from algorithm import colorFilter

# # Etapa de convolucion de la CNN
# def convolution(image_bgr, kernel_size=5, stride=1):
    # """
    # :param image_bgr: imagen original en formato BGR
    # :param kernel_size: size de los patches (kxk pixeles)
    # :param stride: cuantos pixeles se mueve el parche al final de cada analisis 
    # """

    # # Convertir a tensor (batch, height, width, channels)
    # image = tf.expand_dims(image_bgr, axis=0)

    # # Extraer parches
    # patches = tf.image.extract_patches(
        # images=image,
        # sizes=[1, kernel_size, kernel_size, 1],
        # strides=[1, stride, stride, 1],
        # rates=[1, 1, 1, 1],
        # padding='VALID'
    # )

    # patches_np = patches.numpy()
    # out = []

    # # Recorrer cada patch y aplicar segmentRed
    # for row in patches_np[0]:
        # row_out = []
        # for patch_flat in row:
            # patch = patch_flat.reshape(kernel_size, kernel_size, 3)
            # mask = colorFilter.segmentRed(patch)
            # score = np.mean(mask) / 255.0  # Score = que tan "rojo" es el patch
            # row_out.append(score)
        # out.append(row_out)

    # return np.array(out)

# Funcion de convolucion sin utilizar tensorflow (Opcion para RaspberryPi)
def convolution_alt(image_bgr, kernel_size=5, stride=1):
    """
    Aplica una "convolución" manual sobre la imagen usando color rojo como filtro.
    :param image_bgr: Imagen de entrada (H, W, 3)
    :param kernel_size: Tamaño del parche (ej. 5x5)
    :param stride: Paso de la ventana deslizante
    :return: Mapa 2D de scores de "rojez"
    """
    H, W, _ = image_bgr.shape
    out = []

    for y in range(0, H - kernel_size + 1, stride):
        row = []
        for x in range(0, W - kernel_size + 1, stride):
            patch = image_bgr[y:y+kernel_size, x:x+kernel_size]
            mask = colorFilter.segmentRed(patch)  # Devuelve máscara binaria
            score = np.mean(mask) / 255.0  # Normaliza entre 0 y 1
            row.append(score)
        out.append(row)

    return np.array(out)


# Etapa de activacion de la CNN
def activation_relu(x):
    """
    :param x: respuesta de la convolucion 
    """
    return np.maximum(0, x)


# Etapa de pooling de la CNN: funcion max pooling
def pooling(input_array, pool_size=2, stride=2):
    """
    :param input_array: np.array 2D con la salida de la activacion
    :param pool_size: tamaño del parche
    :param stride: cuantos pixeles se mueve el parche
    """
    h, w = input_array.shape # Size del input_array (height, width)

    # Cuanto puede moverse el parche de pooling sin salir de la imagen 
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(0, h - pool_size + 1, stride):
        for j in range(0, w - pool_size + 1, stride):
            patch = input_array[i:i+pool_size, j:j+pool_size]
            output[i//stride, j//stride] = np.max(patch) # Asignar el valor maximo en el parche
    
    return output

# Funcion Softmax para calculo de probabilidades
def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum()


# Etapa de clasificacion de la CNN: funcion softmax
# La imagen se clasifica en un espacio de 9x9 regiones
def classification(input_array):
    """
    :param input_array: np.array 2D con la salida de la etapa de pooling
    """
    h, w = input_array.shape
    h_indices = utils.generate_cell_indices(h, 9) # Obtener indices verticales del grid
    w_indices = utils.generate_cell_indices(w, 9) # Obtener indices horizontales del grid

    classification_output = np.zeros((9, 9), dtype=np.uint8)
    # Para cada espacio del grid 9x9 ...
    for i, (start_h, end_h) in enumerate(h_indices):
        for j, (start_w, end_w) in enumerate(w_indices):
            cell = input_array[start_h:end_h, start_w:end_w]
            score = np.mean(cell)

            # Cortar la celda
            cell = input_array[start_h:end_h, start_w:end_w]
            score = np.mean(cell)

            # Clasificar celda
            # Dos clases: clase 0 (libre), clase 1 (obstaculo)
            logits = np.array([1.0 - score, score])
            probs = softmax(logits)

            # Se toma la clase con mayor probabilidad
            prediction = np.argmax(probs)
            classification_output[i, j] = prediction

    return classification_output

# Funcion para visualizar el resultado de la clasificacion
def viewClassification(img, classified):
    """
    param img: copia de la imagen original con cmap (bgr)
    param classified: resultado de la etapa de clasificacion 
    """
    h, w, _ = img.shape

    # Generar índices para las celdas en la imagen
    h_indices = utils.generate_cell_indices(h, 9)
    w_indices = utils.generate_cell_indices(w, 9)

    # Dibujar las celdas
    for i in range(9):
        for j in range(9):
            start_h, end_h = h_indices[i]
            start_w, end_w = w_indices[j]

            # Ajustar la celda con el valor clasificado
            if classified[i, j] == 1:
                cv2.rectangle(img, (start_w, start_h), (end_w, end_h), (255, 0, 0), 1)  # Azul para obstáculos
            else:
                cv2.rectangle(img, (start_w, start_h), (end_w, end_h), (0, 255, 0), 1)  # Verde para celdas libres

    plt.imshow(img)
    plt.title("Imagen en regiones 9x9 con obstáculos clasificados")
    plt.show()
