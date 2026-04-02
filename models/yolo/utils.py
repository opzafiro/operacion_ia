import numpy as np
import cv2
from config import PROTO_SHAPE, MASK_THRESHOLD, DIST_THRESHOLD

# =========================
# MÁSCARA
# =========================
def generate_mask(coefficients, prototypes):
    """
    Genera la máscara a partir de coeficientes y prototipos.

    Aplica combinación lineal + sigmoid.
    """
    proto_flat = prototypes.reshape(PROTO_SHAPE[0], -1)
    mask = np.dot(coefficients, proto_flat).reshape(PROTO_SHAPE[1:])

    # Función sigmoid
    return 1 / (1 + np.exp(-mask))


def adjust_box_to_original_scale(box, ratio, pad, original_shape):
    """
    Ajusta la bounding box desde el espacio letterbox
    al espacio original de la imagen.
    """
    x1, y1, x2, y2 = box
    pad_w, pad_h = pad
    h, w = original_shape

    # Remover padding y escalar
    x1 = (x1 - pad_w) / ratio
    x2 = (x2 - pad_w) / ratio
    y1 = (y1 - pad_h) / ratio
    y2 = (y2 - pad_h) / ratio

    # Convertir al espacio del proto (160x160)
    x1 = int(x1 / w * PROTO_SHAPE[2])
    x2 = int(x2 / w * PROTO_SHAPE[2])
    y1 = int(y1 / h * PROTO_SHAPE[1])
    y2 = int(y2 / h * PROTO_SHAPE[1])

    # Clamping a límites válidos
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, PROTO_SHAPE[2] - 1), min(y2, PROTO_SHAPE[1] - 1)

    return x1, y1, x2, y2


def extract_object_mask(mask, box_coords, original_size):
    """
    Recorta la máscara en la región de interés y la redimensiona
    al tamaño original de la imagen.
    """
    x1, y1, x2, y2 = box_coords
    h, w = original_size

    # Crear máscara vacía y copiar región de interés
    mask_crop = np.zeros_like(mask)
    mask_crop[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    resized_mask = cv2.resize(mask_crop, (w, h))

    # Binarización
    return (resized_mask > MASK_THRESHOLD).astype(np.uint8)


# =========================
# GEOMETRÍA
# =========================
def get_largest_contour(binary_mask):
    """
    Obtiene el contorno más grande de la máscara binaria.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return max(contours, key=cv2.contourArea) if contours else None


def compute_centroid(contour):
    """
    Calcula el centroide de un contorno usando momentos.
    """
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def compute_direction(cx, cy, width, height):
    """
    Determina la dirección relativa del objeto respecto
    al centro de la imagen.
    """
    center_x = width // 2
    center_y = height // 2

    dx = cx - center_x
    dy = cy - center_y

    distance = np.sqrt(dx**2 + dy**2)

    # Caso: objeto centrado
    if distance < DIST_THRESHOLD:
        return True

    # Determinar dirección dominante
    if abs(dx) > abs(dy):
        return "derecha" if dx > 0 else "izquierda"
    else:
        return "abajo" if dy > 0 else "arriba"
