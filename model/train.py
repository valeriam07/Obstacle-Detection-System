import numpy as np
import cv2
import matplotlib.pyplot as plt
import model, utils

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

            #classify = model.classification(pooled)
            #model.viewClassification(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB), classify)

            #all_preds.append(classify) # Guardar el resultado de la clasificacion (prediccion)

            # Redimensionar la mascara a size del pooling
            resized_mask = cv2.resize(mask_gt, (pooled.shape[1], pooled.shape[0]), interpolation=cv2.INTER_NEAREST)
            grid_gt = utils.convert_mask_to_grid(resized_mask)
            all_gts.append(grid_gt) # Guardar el grid ground truth

            h, w = pooled.shape
            h_indices = utils.generate_cell_indices(h, 9)
            w_indices = utils.generate_cell_indices(w, 9)
            
            image_loss = 0
            prediction_grid = np.zeros((9, 9), dtype=np.uint8)
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

                    prediction_grid[i, j] = np.argmax(probs)

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
            all_preds.append(prediction_grid)
            #model.viewClassification(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB), prediction_grid)
            
        print(f"Epoch {epoch+1}: average loss = {(total_loss/len(dataset)):.4f}")
    print("Entrenamiento finalizado, pesos W: ", W, ", pesos B: ", B)
    np.save("weights.npy", W)
    np.save("biases.npy", B)

image_dir = "./dataset/pexels-110k-512p-min-jpg-depth/images"
mask_dir = "./dataset/pexels_groundTruth"

dataset = utils.load_dataset(image_dir, mask_dir, 300)
train(dataset, epochs=10, lr=0.001)
utils.calculate_metrics(all_preds, all_gts)

