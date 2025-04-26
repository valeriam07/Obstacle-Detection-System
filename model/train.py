import numpy as np
import cv2
import matplotlib.pyplot as plt
import model
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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

            # Normalizar la celda 
            cell_normalized = cell / 255.0  # Normalizar para que los valores estén entre 0 y 1

            # Si más del 10% de la celda tiene obstaculo, se considera clase 1
            if np.mean(cell_normalized) > 0.1:
                grid_gt[i, j] = 1

    return grid_gt

W = np.random.randn(9, 9) * 0.01  # Pesos aleatorios
B = np.zeros((9, 9))  # Sesgos inicializados en 0
all_preds = []
all_gts = []

# Funcion de entrenamiento del modelo CNN
def train(dataset, epochs=5, lr=0.001):
    """
    :param dataset: conjunto de imagenes (original, mascara_gt)
    :param epochs: numero de epochs de entrenamiento
    :param lr: learning rate
    """
    print("Iniciando entrenamiento ...")
    global W, B 

    for epoch in range(epochs):
        total_loss = 0

        for image, mask_gt in dataset:  # imagen y mascara 9x9 ground truth
            conv = model.convolution(image, kernel_size=7, stride=3)
            relu = model.activation_relu(conv)
            pooled = model.pooling(relu, pool_size=2, stride=2)

            if pooled.shape[0] < 9 or pooled.shape[1] < 9:
                print(f"[Info] Imagen omitida por size: {pooled.shape}")
                continue

            classify = model.classification(pooled)
            #model.viewClassification(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB), classify)

            all_preds.append(classify) # Guardar el resultado de la clasificacion (prediccion)

            # Redimensionar la mascara a size del pooling
            resized_mask = cv2.resize(mask_gt, (pooled.shape[1], pooled.shape[0]), interpolation=cv2.INTER_NEAREST)
            grid_gt = convert_mask_to_grid(resized_mask)
            all_gts.append(grid_gt) # Guardar el grid ground truth

            h, w = pooled.shape
            h_indices = model.generate_cell_indices(h, 9)
            w_indices = model.generate_cell_indices(w, 9)
            
            image_loss = 0
            # Para cada celda del grid 9x9...
            for i, (start_h, end_h) in enumerate(h_indices):
                for j, (start_w, end_w) in enumerate(w_indices):

                    cell = pooled[start_h:end_h, start_w:end_w]
                    score = np.mean(cell)

                    # Paso forward
                    z0 = 1.0 - score
                    z1 = score
                    logit = W[i, j] * z1 + B[i, j]
                    probs = model.softmax([z0, logit])

                    # Perdida (cross-entropy)
                    y = grid_gt[i, j]
                    loss = -np.log(probs[y])

                    # Regularización L2 (Ridge)
                    lambda_reg = 0.01  # Coeficiente de regularizacion
                    reg_loss = lambda_reg * np.sum(np.square(W))  # Suma de los cuadrados de los pesos 
                    image_loss += (loss + reg_loss)

                    # Gradientes simples (derivadas parciales)
                    grad = probs.copy()
                    grad[y] -= 1  # derivada de cross-entropy + softmax

                    dW = grad[1] * z1
                    dB = grad[1]

                    # Actualizacion de los pesos 
                    W[i, j] -= lr * dW
                    B[i, j] -= lr * dB

            # Se promedia el loss para las 9x9 celdas en una imagen, correspondiente al loss promedio de esa imagen
            total_loss += image_loss / (9 * 9)  # Promedio de las celdas
            
        print(f"Epoch {epoch+1}: average loss = {(total_loss/len(dataset)):.4f}")
    print("Entrenamiento finalizado, pesos W: ", W, ", pesos B: ", B)
    np.save("weights.npy", W)
    np.save("biases.npy", B)

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


image_dir = "./dataset/pexels-110k-512p-min-jpg-depth/images"
mask_dir = "./dataset/pexels_groundTruth"

dataset = load_dataset(image_dir, mask_dir, 50)
train(dataset, epochs=10, lr=0.001)
calculate_metrics(all_preds, all_gts)

