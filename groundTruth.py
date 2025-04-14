import cv2
import numpy as np
import os

"""
Calculo de la segmentacion de color utilizando funciones de la liberia 
CV2 para generar videos de referencia para el calculo de metricas
"""

# ----------- Rango de rojo HSV ajustado ----------
LOWER_RED1 = np.array([0, 70, 50], np.uint8)
UPPER_RED1 = np.array([10, 255, 255], np.uint8)
LOWER_RED2 = np.array([160, 70, 50], np.uint8)
UPPER_RED2 = np.array([180, 255, 255], np.uint8)

# ----------- Segmentación de color rojo ----------
def segment_color(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    return cv2.bitwise_or(mask1, mask2)

# ----------- Procesar video y guardar solo la máscara ----------
def process_video_mask_only(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Video pero solo la máscara en color
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mask = segment_color(frame)

        # Convierte la máscara (1 canal) a BGR (3 canales)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("Original | Ground Truth", combined)
        out.write(mask_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ----------- Crear carpeta y ejecutar para cada video ----------
os.makedirs("refVideos", exist_ok=True)

input_dir = "testVideos"
output_dir = "refVideos"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mask.avi")
        process_video_mask_only(input_path, output_path)


