import numpy as np
import cv2
import matplotlib.pyplot as plt
import model
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Funcion para generar los indices dividiendo una imagen celdas
def generate_cell_indices(total_size, num_cells):
    """
    :param total_size: size que se quiere dividir en celdas, width o height
    :param num_cells: numero de celdas en las que se quiere dividir
    """
    base = total_size // num_cells
    remainder = total_size % num_cells

    indices = []
    current = 0
    for i in range(num_cells):
        extra = 1 if i < remainder else 0
        start = current
        end = start + base + extra
        indices.append((start, end))
        current = end
    return indices

def resize_image(image, target_height):
    # Obtener las dimensiones originales de la imagen
    original_height, original_width = image.shape[:2]

    # Calcular el factor de escala manteniendo la proporción
    aspect_ratio = original_width / original_height
    target_width = int(target_height * aspect_ratio)

    # Redimensionar la imagen manteniendo la proporción
    resized_image = cv2.resize(image, (target_width, target_height))

    return resized_image

# Cargar dataset para entrenamiento
def load_dataset(image_dir, mask_dir, limit=5): 
    """
    :param image_dir: directorio de las imagenes de entrenamiento originales
    :param mask_dir: directorio del ground truth de la segmentacion 
    :param limit: cantidad maximo de imagenes que cargar
    """
    dataset = []
    image_filenames = sorted(os.listdir(image_dir))[:limit]

    for filename in image_filenames:
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + ".png")

        # Cargar imagen y mascara
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask_gray is None:
            print(f"Saltando archivo: {filename}")
            continue

        # Normalizar imagen de profundidad a rango 0–255
        depth_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Aplicar colormap YlOrRd
        cmap = plt.get_cmap("YlOrRd")
        depth_colored = cmap(depth_norm)[:, :, :3] 
        depth_colored_rgb = (depth_colored * 255).astype(np.uint8)
        depth_colored_bgr = cv2.cvtColor(depth_colored_rgb, cv2.COLOR_RGB2BGR)

        # Redimensionar la imagen manteniendo la proporción
        resized_image = resize_image(depth_colored_bgr, 99)

        # Redimensionar la máscara a las dimensiones de la imagen redimensionada
        resized_mask = cv2.resize(mask_gray, (resized_image.shape[1], resized_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Agregar al dataset
        dataset.append((resized_image, resized_mask))

    print(f"Dataset cargado con {len(dataset)} elementos.")
    return dataset

# Convertir una mascara binaria (ground truth) en un grid 9x9 
def convert_mask_to_grid(mask, target_shape=(9, 9)):
    """
    :param mask: mascara binaria
    :target_shape: forma del grid (9x9)
    """
    h, w = mask.shape
    gh, gw = target_shape

    cell_h = h // gh
    cell_w = w // gw
    extra_h = h % gh
    extra_w = w % gw

    grid_gt = np.zeros((gh, gw), dtype=np.uint8)

    for i in range(gh):
        for j in range(gw):
            start_h = i * cell_h + min(i, extra_h)
            end_h = (i + 1) * cell_h + min(i + 1, extra_h)
            start_w = j * cell_w + min(j, extra_w)
            end_w = (j + 1) * cell_w + min(j + 1, extra_w)

            cell = mask[start_h:end_h, start_w:end_w]

            white_ratio = np.count_nonzero(cell == 255) / cell.size

            # Si más del 10% de la celda tiene blanco (obstaculo segmentado), se considera clase 1
            if white_ratio > 0.1:
                grid_gt[i, j] = 1
    return grid_gt

# Funcion para calcular metricas del entrenamiento (SKLearn)
def calculate_metrics(all_preds, all_gts):
    """
    :param all_preds: Lista de predicciones del modelo (9x9 grid)
    :param all_gts: Lista de etiquetas ground truth (9x9 grid)
    :return: Tuple de métricas (precisión, recall, F1, exactitud)
    """
    # Aplanar las listas de predicciones y ground truths
    flattened_preds = np.concatenate([pred.flatten() for pred in all_preds])
    flattened_gts = np.concatenate([gt.flatten() for gt in all_gts])

    # Calcular precisión, recall, F1 y exactitud
    accuracy = accuracy_score(flattened_gts, flattened_preds)
    precision = precision_score(flattened_gts, flattened_preds)
    recall = recall_score(flattened_gts, flattened_preds)
    f1 = f1_score(flattened_gts, flattened_preds)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    return accuracy, precision, recall, f1

