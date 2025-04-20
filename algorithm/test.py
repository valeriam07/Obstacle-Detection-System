import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Rutas
input_dir = "pexels-110k-512p-min-jpg-depth/images"  # carpeta con las imágenes de profundidad
output_dir = "pexels_groundTruth"  # carpeta para guardar las máscaras binarias

# Crear la carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Obtener lista de imágenes JPEG en la carpeta
image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpeg")]

# Colormap invertido (zonas cercanas en rojo)
cmap = plt.get_cmap("YlOrRd")

# Procesar cada imagen
for filename in tqdm(image_files, desc="Procesando imágenes"):
    # Cargar imagen de profundidad
    depth = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_UNCHANGED)
    
    if depth is None:
        continue

    # Normalizar a 0–255
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Aplicar colormap YlOrRd
    depth_colored = cmap(depth_norm)[:, :, :3]  # quitar canal alfa
    depth_colored_rgb = (depth_colored * 255).astype(np.uint8)

    # Crear máscara para zonas rojas (en formato RGB)
    # Ajusta los rangos si es necesario para tu colormap
    red_mask = cv2.inRange(depth_colored_rgb, (150, 0, 0), (255, 110, 110))

    # Guardar máscara binaria como imagen PNG
    output_path = os.path.join(output_dir, filename.replace(".jpeg", ".png"))
    cv2.imwrite(output_path, red_mask)
