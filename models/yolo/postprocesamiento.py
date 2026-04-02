import numpy as np
from config import CONF_THRESHOLD



def compute_iou(box, boxes):
    """
    Calcula el Intersection over Union (IoU) entre una caja y múltiples cajas.

    Args:
        box (np.ndarray): Caja individual [x1, y1, x2, y2].
        boxes (np.ndarray): Array de cajas.

    Returns:
        np.ndarray: IoU de cada caja.
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - intersection
    return intersection / (union + 1e-6)


def non_max_suppression(boxes, scores, threshold):
    """
    Aplica Non-Maximum Suppression para eliminar detecciones redundantes.

    Args:
        boxes (np.ndarray): Cajas detectadas.
        scores (np.ndarray): Confianzas.
        threshold (float): Umbral IoU.

    Returns:
        list: Índices de cajas seleccionadas.
    """
    sorted_indices = np.argsort(scores)[::-1]
    selected_indices = []

    while len(sorted_indices) > 0:
        current = sorted_indices[0]
        selected_indices.append(current)

        if len(sorted_indices) == 1:
            break

        ious = compute_iou(boxes[current], boxes[sorted_indices[1:]])
        sorted_indices = sorted_indices[1:][ious < threshold]

    return selected_indices


def filter_detections(boxes, scores, mask_coeffs):
    """
    Filtra detecciones según el umbral de confianza.

    Returns:
        boxes, scores, mask_coeffs filtrados
    """
    valid_mask = scores > CONF_THRESHOLD
    return boxes[valid_mask], scores[valid_mask], mask_coeffs[valid_mask]


def get_best_detection(boxes, scores, mask_coeffs):
    """
    Selecciona la detección con mayor score.
    """
    best_index = np.argmax(scores)
    return boxes[best_index], mask_coeffs[best_index]