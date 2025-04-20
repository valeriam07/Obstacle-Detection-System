import cv2
import numpy as np
import colorFilter


#___________________________________/ Prueba con imagenes /__________________________________
def testImage():
    # Cargar imagen (relative path)
    image = cv2.imread("./samples/testImages/rose.jpg")

    # Aplicar la segmentacion de color
    mask = colorFilter.segmentRed(image)

    # Convertir a escala de grises
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Combinar imagenes
    combined = np.hstack((image, mask_colored))

    # Visualizacion de las imagenes (original y procesada)
    scale_percent = 70  # Escalar al 50% del tamaño original
    width = int(combined.shape[1] * scale_percent / 100)
    height = int(combined.shape[0] * scale_percent / 100)
    combined_resized = cv2.resize(combined, (width, height), interpolation=cv2.INTER_AREA)

    # Mostrar imagenes
    cv2.imshow("Imagen Original | Máscara Roja", combined_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#__________________________/ Prueba con video /________________________________________

def testVideo():
    cap = cv2.VideoCapture("./samples/testVideos/rose.mp4")  

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara o el archivo de video.")
        exit()

    predicted_masks = []
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))  # bajar calidad (320x240) para mas velocidad 

        if not ret:
            print("⚠️ Fin del video o error en la captura.")
            break

        # Segmentación del color rojo
        mask = colorFilter.segmentRed(frame)
        mask = colorFilter.filterByColorDensity(mask)
        predicted_masks.append(mask)

        # -- Visualizar obstaculos --
        # Encontrar los contornos del obstáculo en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujar un rectángulo alrededor de cada contorno detectado
        for contour in contours:
            if cv2.contourArea(contour) > 700: 
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)  
        
        
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Visualización comparativa
        combined = np.hstack((frame, mask_colored))
        combined_resized = cv2.resize(combined, (800, 400))  

        cv2.imshow("Obstaculos detectados | Segmentación Roja", combined_resized)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


    # --- Calculo de metricas de la mascara obtenida ---
    metrics = colorFilter.metrics("./samples/refVideos/rose_mask.avi", predicted_masks)
    precision, recall, f1, iou = metrics
    print("\nMETRICAS GENERALES:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")

testVideo()