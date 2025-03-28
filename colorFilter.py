import cv2
import numpy as np

# Ecualizacion de histograma de un canal (h, s, v) 
# Funcion de CV2: cv2.equalizeHist()
def histogram_equalization(channel):
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


# Segmentar una imagen por color (rojo)
def segmentRed(image):
    # Convertir al espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Preprocesamiento: normalizar y ecualizar
    h, s, v = cv2.split(hsv)
    v = histogram_equalization(v)  # Ecualizacion del canal V (brillo)
    enhanced_hsv = cv2.merge([h, s, v]) # Recombinar los tres canales del espacio de color HSV

    # Rangos de color rojo HSV (dos rangos de rojos) [Falta verificar el rango del rojo de las imagenes reales]
    lower_red1 = np.array([0, 120, 70], np.uint8)   # Rojo oscuro
    upper_red1 = np.array([10, 255, 255], np.uint8)

    lower_red2 = np.array([170, 120, 70], np.uint8) # Rojo brillante
    upper_red2 = np.array([180, 255, 255], np.uint8)

    # Create red masks and combine them
    mask1 = colorMask(enhanced_hsv, lower_red1, upper_red1)
    mask2 = colorMask(enhanced_hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2) # Combinar las mascaras de los dos tonos de rojo

    return red_mask

# Cargar imagen (relative path)
image = cv2.imread("testImages/rose.jpg")

# Aplicar la segmentacion de color
mask = segmentRed(image)

# Convertir a escala de grises
mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Combinar imagenes
combined = np.hstack((image, mask_colored))

# Visualizacion de las imagenes (original y procesada)
scale_percent = 50  # Escalar al 50% del tamaño original
width = int(combined.shape[1] * scale_percent / 100)
height = int(combined.shape[0] * scale_percent / 100)
combined_resized = cv2.resize(combined, (width, height), interpolation=cv2.INTER_AREA)

# Mostrar imagenes
cv2.imshow("Imagen Original | Máscara Roja", combined_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()