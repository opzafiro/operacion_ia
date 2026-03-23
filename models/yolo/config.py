
#=====================
#   YOLO
#=====================

YOLO_MODEL_PATH = "models/yolo/YOLO26n-seg-200.onnx"


CONF_THRESHOLD = 0.25     # Umbral mínimo de confianza para detecciones
IOU_THRESHOLD = 0.45      # Umbral de IoU para NMS
DIST_THRESHOLD = 30       # Distancia mínima para considerar centrado

PROTO_SHAPE = (32, 160, 160)  # (canales, alto, ancho) del proto de máscaras
MASK_THRESHOLD = 0.5          # Umbral binarización máscara