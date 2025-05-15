import cv2
import numpy as np
import time
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.append(parent_dir)
import model,utils

import ArducamDepthCamera as ac  

MAX_DISTANCE = 4000  # Distancia máxima de detección en mm
GRID_SIZE = 9        # Grid 9x9

# Forma de visualizar las zonas detectadas como obstaculo dentro del grid de la imagen
def draw_grid_overlay(image, prediction_grid):
    h, w, _ = image.shape
    cell_h = h // GRID_SIZE
    cell_w = w // GRID_SIZE

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w

            color = (0, 255, 0)  # Verde por defecto
            if prediction_grid[i, j] == 1:
                color = (0, 0, 255)  # Rojo si hay obstáculo

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    return image

# Forma de visualizar los obstaculos detectados: marca el area donde hay obstaculo
def draw_obstacle_overlay(image, prediction_grid, obstacle_distances):
    h, w, _ = image.shape
    cell_h = h // GRID_SIZE
    cell_w = w // GRID_SIZE
    obstacle_rectangles = []

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if prediction_grid[i, j] == 1:  # Si hay obstaculo
                y1 = i * cell_h
                y2 = (i + 1) * cell_h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w
                obstacle_rectangles.append((x1, y1, x2, y2))

                # Dibujar rectangulo celeste
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 1)

                # Mostrar distancia de las celdas con obstaculo
                if obstacle_distances[i,j] >= 0:
                    dist = obstacle_distances[i, j]
                    cv2.putText(
                        image,
                        f"{dist:.2f} m",
                        (x1 + 2, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.2,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
    return image

def main():
    print("Inicializando camara ToF...")
    cam = ac.ArducamCamera()
    ret = cam.open(ac.Connection.CSI, 0)
    if ret != 0:
        print("Error al abrir la camara:", ret)
        return

    ret = cam.start(ac.FrameType.DEPTH)
    if ret != 0:
        print("Error al iniciar la camara:", ret)
        cam.close()
        return
    
    cam.setControl(ac.Control.RANGE, MAX_DISTANCE)
    r = cam.getControl(ac.Control.RANGE)

    print("Cargando pesos...")
    W = np.load("../model/saved_models/v4/weights.npy")
    B = np.load("../model/saved_models/v4/biases.npy")

    print("Presiona 'q' para salir.")
    
    prev_time = time.time()
    
    out_path = "output/Arducam_out.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, 8, (800, 600))
    
    while True:
        frame = cam.requestFrame(2000)
        
        if frame and isinstance(frame, ac.DepthData):
            depth_buf = frame.depth_data
            depth_meters = depth_buf / 1000.0  # convertir a metros
            depth_image = (depth_buf * (255.0 / r)).astype(np.uint8)
            colorized = cv2.applyColorMap(depth_image, cv2.COLORMAP_RAINBOW)
            
            colorized_resized = cv2.resize(colorized, (120, 90))  #(width,height), original shape = (height=240, width=180)

            # Predecir grid 9x9
            conv = model.convolution_alt(colorized_resized, kernel_size=3, stride=5)
            relu = model.activation_relu(conv)
            pooled = model.pooling(relu, pool_size=2, stride=2)

            if pooled.shape[0] < GRID_SIZE or pooled.shape[1] < GRID_SIZE:
                print("[Info] Frame omitido por tamaño pequeño.")
                cam.releaseFrame(frame)
                continue

            h_idx = utils.generate_cell_indices(pooled.shape[0], GRID_SIZE)
            w_idx = utils.generate_cell_indices(pooled.shape[1], GRID_SIZE)
            prediction_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            
            obstacle_distances = np.full((GRID_SIZE, GRID_SIZE), -1.0)
            
            for i, (sh, eh) in enumerate(h_idx):
                for j, (sw, ew) in enumerate(w_idx):
                    cell = pooled[sh:eh, sw:ew]
                    score = np.mean(cell)
                    z0 = 1.0 - score
                    z1 = score
                    logit = W[i, j] * z1 + B[i, j]
                    probs = model.softmax([z0, logit])
                    pred = np.argmax(probs)
                    prediction_grid[i, j] = pred
                    
                    if pred == 1:  # 1 indica obstaculo
                        # Extraer distancia promedio de la celda original
                        depth_cell = depth_meters[sh*5:eh*5, sw*5:ew*5]  # Escalar a la imagen original
                        distance = np.mean(depth_cell)
                        obstacle_distances[i, j] = distance

            # Dibujar celdas
            result = draw_obstacle_overlay(colorized.copy(), prediction_grid, obstacle_distances)
            
            # Capturar y guardar resultado
            out.write(result)
            
            # Medir FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Mostrar FPS en la imagen
            cv2.putText(result, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


            # Mostrar en ventana
            cv2.namedWindow("ToF Grid Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ToF Grid Detection", 800, 600)  
            cv2.imshow("ToF Grid Detection", result)
            cam.releaseFrame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                out.release()
                break

    cam.stop()
    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
