import cv2
import numpy as np


IMG_SIZE = 640 


def resize_with_letterbox(image, target_size=IMG_SIZE, color=(114, 114, 114)):
    """
    Redimensiona una imagen manteniendo la relación de aspecto (letterbox),
    agregando padding para ajustarla a un tamaño cuadrado.

    Args:
        image (np.ndarray): Imagen original (BGR).
        target_size (int): Tamaño objetivo (cuadrado).
        color (tuple): Color del padding.

    Returns:
        padded (np.ndarray): Imagen redimensionada con padding.
        scale_ratio (float): Factor de escala aplicado.
        (pad_w, pad_h) (tuple): Padding aplicado en ancho y alto.
    """
    original_shape = image.shape[:2]

    # Factor de escala manteniendo aspecto
    scale_ratio = min(target_size / original_shape[0], target_size / original_shape[1])

    resized_shape = (
        int(round(original_shape[1] * scale_ratio)),
        int(round(original_shape[0] * scale_ratio)),
    )

    # Cálculo de padding
    pad_w = (target_size - resized_shape[0]) / 2
    pad_h = (target_size - resized_shape[1]) / 2

    # Redimensionar imagen
    resized = cv2.resize(image, resized_shape, interpolation=cv2.INTER_LINEAR)

    # Bordes (ajuste fino con 0.1 para evitar errores de redondeo)
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return padded, scale_ratio, (pad_w, pad_h)


def normalize_image(image):
    """
    Convierte la imagen a formato adecuado para el modelo:
    - BGR → RGB
    - Normalización [0,1]
    - Formato CHW
    - Añade dimensión batch

    Args:
        image (np.ndarray): Imagen en BGR.

    Returns:
        np.ndarray: Tensor listo para inferencia.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    image_chw = np.transpose(image_norm, (2, 0, 1))
    return np.expand_dims(image_chw, axis=0)