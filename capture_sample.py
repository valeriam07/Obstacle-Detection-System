import cv2
import numpy as np
import ArducamDepthCamera as ac
import time
import os
import re

MAX_DISTANCE = 4000
SAVE_FOLDER = "Arducam_samples"
VIDEO_DURATION = 10  # segundos
FPS = 15  # cuadros por segundo

def get_next_video_number(folder):
    os.makedirs(folder, exist_ok=True)
    existing_files = os.listdir(folder)
    numbers = []
    for name in existing_files:
        match = re.match(r"arducam_(\d+)\.avi", name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=-1) + 1

def main():
    print("Arducam Depth Camera Video Capture")
    print("  SDK version:", ac.__version__)

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
    info = cam.getCameraInfo()
    width, height = info.width, info.height

    # Obtener nombre de archivo
    video_number = get_next_video_number(SAVE_FOLDER)
    out_path = os.path.join(SAVE_FOLDER, f"arducam_{video_number}.avi")

    print("Esperando 5 segundos antes de grabar... puedes ver la vista previa en vivo")
    start_preview_time = time.time()
    while time.time() - start_preview_time < 5:
        frame = cam.requestFrame(2000)
        if frame and isinstance(frame, ac.DepthData):
            depth_buf = frame.depth_data
            result_image = (depth_buf * (255.0 / r)).astype(np.uint8)
            result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_RAINBOW)
            cv2.imshow("Vista previa en vivo", result_image)
            cam.releaseFrame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Cancelado")
                cam.stop()
                cam.close()
                cv2.destroyAllWindows()
                return

    # Crear el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, FPS, (width, height))

    print(f"Grabando video: {out_path}")
    start_time = time.time()
    while time.time() - start_time < VIDEO_DURATION:
        frame = cam.requestFrame(2000)
        if frame and isinstance(frame, ac.DepthData):
            depth_buf = frame.depth_data
            result_image = (depth_buf * (255.0 / r)).astype(np.uint8)
            result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_RAINBOW)

            out.write(result_image)
            cv2.imshow("Vista previa", result_image)

            cam.releaseFrame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Grabacion interrumpida")
                break

    print("Video guardado correctamente en:", out_path)
    out.release()
    cv2.destroyAllWindows()
    cam.stop()
    cam.close()

if __name__ == "__main__":
    main()
