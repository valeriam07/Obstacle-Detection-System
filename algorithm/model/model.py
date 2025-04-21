import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.append("./algorithm")
import colorFilter

# Etapa de convolucion de la CNN
def convolution(image_bgr, kernel_size=5, stride=1):
    """
    :param image_bgr: imagen original en formato BGR
    :param kernel_size: size de los patches (kxk pixeles)
    :param stride: cuantos pixeles se mueve el parche al final de cada analisis 
    """

    # Convertir a tensor (batch, height, width, channels)
    image = tf.expand_dims(image_bgr, axis=0)

    # Extraer parches
    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    patches_np = patches.numpy()
    out = []

    # Recorrer cada patch y aplicar segmentRed
    for row in patches_np[0]:
        row_out = []
        for patch_flat in row:
            patch = patch_flat.reshape(kernel_size, kernel_size, 3)
            mask = colorFilter.segmentRed(patch)
            score = np.mean(mask) / 255.0  # Score = que tan "rojo" es el patch
            row_out.append(score)
        out.append(row_out)

    return np.array(out)

# Etapa de activacion de la CNN
def activation_relu(x):
    """
    :param x: respuesta de la convolucion 
    """
    return np.maximum(0, x)



# --------------------- Cargar imagen ---------------------------------------

img = cv2.imread("./dataset/pexels-110k-512p-min-jpg-depth/images/depth-1000171.jpeg", cv2.IMREAD_UNCHANGED)

# Normalizar a 0–255
depth_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Aplicar colormap YlOrRd (mas parecido a imagenes de Arducam)
cmap = plt.get_cmap("YlOrRd")
depth_colored = cmap(depth_norm)[:, :, :3] 
depth_colored_rgb = (depth_colored * 255).astype(np.uint8)
depth_colored_bgr = cv2.cvtColor(depth_colored_rgb, cv2.COLOR_RGB2BGR)

# plt.imshow(depth_colored_rgb)
# plt.title("Imagen original como colormap")
# plt.show()

#------------------------ Aplicar Convolucion: Que tan roja es x region? --------------------
response = convolution(depth_colored_bgr, kernel_size=7, stride=3)

plt.imshow(response)
plt.title("Respuesta tipo convolución usando segmentRed")
plt.colorbar()
plt.show()

#-------------------- Aplicar Activacion: Activacion de las neuronas segun ------------
# ------------------- la suma ponderada de las entradas                    ------------
activated = activation_relu(response)

plt.imshow(activated, cmap='gray')
plt.title("Activación (umbral > 0.4)")
plt.colorbar()
plt.show()
