import cv2
import numpy as np
import colorFilter


#___________________________________/ Prueba con imagenes /__________________________________
def testImage():
    # Cargar imagen (relative path)
    image = cv2.imread("testImages/tof_img1.jpg")

    # Aplicar la segmentacion de color
    mask = colorFilter.segmentRed(image)

    # Convertir a escala de grises
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Combinar imagenes
    combined = np.hstack((image, mask_colored))

    # Visualizacion de las imagenes (original y procesada)
    scale_percent = 100  # Escalar al 50% del tamaño original
    width = int(combined.shape[1] * scale_percent / 100)
    height = int(combined.shape[0] * scale_percent / 100)
    combined_resized = cv2.resize(combined, (width, height), interpolation=cv2.INTER_AREA)

    # Mostrar imagenes
    cv2.imshow("Imagen Original | Máscara Roja", combined_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#__________________________/ Prueba con video /________________________________________

def testVideo():
    cap = cv2.VideoCapture("testVideos/colorSorting2.mp4")  

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara o el archivo de video.")
        exit()

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))  # bajar calidad (320x240) para mas velocidad 
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        if not ret:
            print("⚠️ Fin del video o error en la captura.")
            break

        # Segmentación del color rojo
        mask = colorFilter.segmentRed(frame)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Visualización
        combined = np.hstack((frame, mask_colored))
        combined_resized = cv2.resize(combined, (800, 400))  

        cv2.imshow("Video Original | Segmentación Roja", combined_resized)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

testVideo()