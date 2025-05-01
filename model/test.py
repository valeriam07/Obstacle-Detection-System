import numpy as np
import cv2
import matplotlib.pyplot as plt
import model, utils
import os

# Modulo de prediccion CNN a partir del modelo entrenado
def predict(test_images, weights_path="weights.npy", bias_path="biases.npy", limit=10):
    """
    Realiza la predicción sobre nuevas imagenes usando los pesos entrenados
    :param test_images: Directorio de imágenes a predecir
    :param weights_path: Ruta al archivo .npy de pesos W
    :param bias_path: Ruta al archivo .npy de sesgos B
    :param limit: Numero maximo de imagenes a procesar
    """
    # Cargar pesos y sesgos
    W = np.load(weights_path)
    B = np.load(bias_path)

    all_preds = []
    for img in test_images:

        # Aplicar modelo 
        conv = model.convolution(img, kernel_size=7, stride=3)
        relu = model.activation_relu(conv)
        pooled = model.pooling(relu, pool_size=2, stride=2)

        if pooled.shape[0] < 9 or pooled.shape[1] < 9:
            print(f"[Info] Imagen omitida por size: {pooled.shape}")
            continue

        h, w = pooled.shape
        h_indices = utils.generate_cell_indices(h, 9)
        w_indices = utils.generate_cell_indices(w, 9)

        prediction_grid = np.zeros((9, 9), dtype=np.uint8)

        # Calcular predicciones (grid 9x9)
        for i, (start_h, end_h) in enumerate(h_indices):
            for j, (start_w, end_w) in enumerate(w_indices):
                cell = pooled[start_h:end_h, start_w:end_w]
                score = np.mean(cell)

                z0 = 1.0 - score
                z1 = score
                logit = W[i, j] * z1 + B[i, j]
                probs = model.softmax([z0, logit])

                prediction_grid[i, j] = np.argmax(probs)

        # Guardar prediccion de la imagen
        all_preds.append(prediction_grid)
        # Visualizar
        model.viewClassification(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB), prediction_grid)
    return all_preds

# Evaluar predicciones 
def evaluate_prediction(masks, preds):
    all_gts = []
    for mask, pred in zip(masks, preds): 
        # Calcular los grids del ground truth
        grid_gt = utils.convert_mask_to_grid(mask)
        print(f"Ground truth grid:\n{grid_gt}")
        all_gts.append(grid_gt)  
        
        # Visualizar ground truth
        model.viewClassification(cv2.cvtColor(mask.copy(), cv2.COLOR_BGR2RGB), grid_gt)

    # Calcular las metricas de la prediccion 
    utils.calculate_metrics(all_gts, preds)



image_dir = "./dataset/pexels_test"
mask_dir = "./dataset/pexels_groundTruth"

dataset = utils.load_dataset(image_dir, mask_dir, 1)
imgs, masks = zip(*dataset)

imgs = list(imgs)
masks = list(masks)

preds = predict(imgs, limit=5)
evaluate_prediction(masks, preds)

