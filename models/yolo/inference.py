import cv2
import numpy as np
import onnxruntime as ort
from .preprocesamiento import *
from .postprocesamiento import *
from .utils import *
from .config import YOLO_MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, DIST_THRESHOLD, PROTO_SHAPE, MASK_THRESHOLD 


# =========================
# SESIÓN ONNX
# =========================
# Inicializa la sesión de inferencia una sola vez (mejor rendimiento)
session = ort.InferenceSession(YOLO_MODEL_PATH)
input_name = session.get_inputs()[0].name


def infer_direction(image: np.ndarray):
    """
    Pipeline completo:
    - Preprocesamiento
    - Inferencia ONNX
    - Postprocesamiento

    Args:
        image (np.ndarray): Imagen de entrada (BGR)

    Returns:
        str | bool | None:
            - cx, cy, width, height, contour
    """
    height, width = image.shape[:2]

    # --- Preprocesamiento ---
    image_lb, ratio, pad = resize_with_letterbox(image)
    input_tensor = normalize_image(image_lb)

    # --- Inferencia ---
    output0, output1 = session.run(None, {input_name: input_tensor})

    predictions = output0[0]
    prototypes = output1[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    mask_coeffs = predictions[:, 6:]

    # --- Filtrado ---
    boxes, scores, mask_coeffs = filter_detections(boxes, scores, mask_coeffs)
    if len(boxes) == 0:
        return None

    # --- NMS ---
    keep = non_max_suppression(boxes, scores, IOU_THRESHOLD)
    boxes, scores, mask_coeffs = boxes[keep], scores[keep], mask_coeffs[keep]

    # --- Mejor detección ---
    box, coeff = get_best_detection(boxes, scores, mask_coeffs)

    # --- Máscara ---
    mask = generate_mask(coeff, prototypes)
    adjusted_box = adjust_box_to_original_scale(box, ratio, pad, (height, width))
    binary_mask = extract_object_mask(mask, adjusted_box, (height, width))

    # --- Contornos ---
    contour = get_largest_contour(binary_mask)
    if contour is None:
        return None

    # --- Centroide ---
    centroid = compute_centroid(contour)
    if centroid is None:
        return None

    cx, cy = centroid

    return (cx, cy, width, height, contour, compute_direction(cx, cy, width, height))


# =========================
# EJECUCIÓN
# =========================
if __name__ == "__main__":
    image = cv2.imread("images/prueba/input1.jpeg")
    direction = infer_direction(image)
    print(direction)