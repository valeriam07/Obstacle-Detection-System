import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import model

# --------------------- Cargar imagen ---------------------------------------

img = cv2.imread("./dataset/pexels-110k-512p-min-jpg-depth/images/depth-1000129.jpeg", cv2.IMREAD_UNCHANGED)

# Normalizar a 0–255
depth_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Aplicar colormap YlOrRd (mas parecido a imagenes de Arducam)
cmap = plt.get_cmap("YlOrRd")
depth_colored = cmap(depth_norm)[:, :, :3] 
depth_colored_rgb = (depth_colored * 255).astype(np.uint8)
depth_colored_bgr = cv2.cvtColor(depth_colored_rgb, cv2.COLOR_RGB2BGR)

plt.imshow(depth_colored_rgb)
plt.title("Imagen original con colormap")
plt.show()


#------------------------ Aplicar Convolucion: Que tan roja es x region? --------------------
response = model.convolution(depth_colored_bgr, kernel_size=7, stride=3)

plt.imshow(response)
plt.title("Respuesta tipo convolución usando segmentRed")
plt.colorbar()
plt.show()

#-------------------- Aplicar Activacion: Activacion de las neuronas segun ------------
# ------------------- la suma ponderada de las entradas                    ------------
activated = model.activation_relu(response)

plt.imshow(activated, cmap='gray')
plt.title("Activación")
plt.colorbar()
plt.show()

#------------------ Aplicar Pooling: Agrupar informacion de la imagen -----------------
pooled = model.pooling(activated, pool_size=2, stride=2)

im2 = plt.imshow(pooled, cmap='hot')
plt.title("Pooling")
plt.axis("off")
plt.colorbar()
plt.show()

#------------------ Aplicar Clasificacion: Clasificar la imagen por regiones ------------- 
#------------------ grid 9x9: 0 = no hay obstaculo, 1 = hay obstaculo        -------------
classified = model.classification(pooled)
img = depth_colored_rgb.copy()
model.viewClassification(img, classified)