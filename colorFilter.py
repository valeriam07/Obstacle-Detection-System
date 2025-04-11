import cv2
import numpy as np

# Ecualizacion de histograma de un canal (h, s, v) 
# Funcion de CV2: cv2.equalizeHist()
def histogram_equalization(channel):
    """
    :param channel: Canal H, S o V que se quiere ecualizar
    """
    # Calcular histograma del canal(256 niveles)
    hist, bins = np.histogram(channel.flatten(), 256, [0,256])
    
    # Obtener la funcion de distribucion acumulativa (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()  # Normalización (0-1)

    # Escalar la CDF a 0-255
    cdf_scaled = (cdf_normalized * 255).astype(np.uint8)

    # Reasignar valores usando la CDF escalada
    equalized_channel = cdf_scaled[channel]

    return equalized_channel

# Crea una máscara binaria para detectar un color especifico en una imagen en espacio HSV
# Funcion CV2: cv2.inRange()
def colorMask(hsv_image, lower_bound, upper_bound):
    """
    :param hsv_image: Imagen en espacio de color HSV
    :param lower_bound: Limite inferior del color (H, S, V)
    :param upper_bound: Limite superior del color (H, S, V)
    """
    
    # Dividir la imagen HSV en sus componentes H, S, V
    h, s, v = cv2.split(hsv_image)

    # Obtener las dimensiones de la imagen
    height, width = hsv_image.shape[:2]

    # Crear una mascara de ceros (negro) para la imagen binaria
    mask = np.zeros((height, width), dtype=np.uint8)

    # Recorrer cada pixel de la imagen
    for y in range(height):
        for x in range(width):
            # Obtener el valor H, S, V de cada píxel
            h_val = h[y, x]
            s_val = s[y, x]
            v_val = v[y, x]

            # Verificar si el píxel está dentro del rango del color 
            if (lower_bound[0] <= h_val <= upper_bound[0] and
                lower_bound[1] <= s_val <= upper_bound[1] and
                lower_bound[2] <= v_val <= upper_bound[2]):
                mask[y, x] = 255  # El pixel dentro del rango, se pone blanco

    return mask # Mascara binaria donde los píxeles dentro del rango de color son 255 y los demás son 0

# Verifica la densidad del area de color detectada, descarta aquellas bajo un umbral
def filterByColorDensity(mask):
    """
    param mask: Imagen tipo matriz (MatLike), aplicada la mascara de color
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    solid_mask = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 700:  # Ignorar objetos muy pequeños
            # Crear una máscara del contorno
            mask_contour = np.zeros_like(mask)
            cv2.drawContours(mask_contour, [contour], -1, (255), thickness=cv2.FILLED)

            # Calcular el porcentaje de píxeles dentro de la máscara de color
            color_area = cv2.bitwise_and(mask_contour, mask)
            color_percentage = np.sum(color_area == 255) / np.sum(mask_contour == 255)

            # Filtrar objetos con baja densidad de color
            if color_percentage > 0.85:  # Umbral de consistencia de color
                solid_mask = cv2.drawContours(solid_mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Operaciones morfológicas para mejorar la segmentación
    kernel = np.ones((5, 5), np.uint8)
    solid_mask = cv2.dilate(solid_mask, kernel, iterations=1)  # Dilatar para unir áreas
    solid_mask = cv2.erode(solid_mask, kernel, iterations=1)  # Erosionar para eliminar ruido

    return solid_mask

# Segmentar una imagen por color (rojo)
def segmentRed(image):
    """
    param image: imagen 
    """
    # Convertir al espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Preprocesamiento: normalizar y ecualizar
    h, s, v = cv2.split(hsv)
    s = histogram_equalization(s)  # Ecualización del canal S
    v = histogram_equalization(v)  # Ecualizacion del canal V (brillo)
    enhanced_hsv = cv2.merge([h, s, v]) # Recombinar los tres canales del espacio de color HSV

    # Rangos de rojo HSV utiles para entornos reales 
    lower_red1 = np.array([0, 70, 50], np.uint8)    # Rojo puro con saturación mínima y valor mínimo
    upper_red1 = np.array([10, 255, 255], np.uint8)  # Rojo claro, sin llegar al naranja

    lower_red2 = np.array([160, 70, 50], np.uint8)
    upper_red2 = np.array([180, 255, 255], np.uint8)

    # Rangos de rojo HSV utiles para imagenes mapa de color 
    # lower_red1 = np.array([0, 0, 0], np.uint8)
    # upper_red1 = np.array([10, 255, 255], np.uint8)

    # # Rango 2 – Rojo apagado brillante/magenta
    # lower_red2 = np.array([160, 20, 20], np.uint8)
    # upper_red2 = np.array([180, 255, 255], np.uint8)

    # Crear mascaras rojas y combinarlas
    mask1 = colorMask(enhanced_hsv, lower_red1, upper_red1)
    mask2 = colorMask(enhanced_hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2) # Combinar las mascaras de los dos tonos de rojo

    return red_mask

