from picamera2 import Picamera2
from libcamera import controls
import onnxruntime as ort
import numpy as np
from preprocesamiento import resize_with_letterbox, normalize_image
from postprocesamiento import filter_detections, non_max_suppression, get_best_detection
from utils import generate_mask, adjust_box_to_original_scale, extract_object_mask
from config import YOLO_MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, DIST_THRESHOLD

class Camera:
    def __init__(self):
        self.picam2 = Picamera2()
        self.configure_preview = self.picam2.create_preview_configuration(main={"format": 'RGB888', "size": (900, 600)})
        self.configure_still = self.picam2.create_still_configuration(main={"format": 'RGB888', "size": (9000, 6000)})
        self.picam2.configure(self.configure_preview)
        self.picam2.start()

    def autofocus(self):
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
        success = self.picam2.autofocus_cycle()
        return success
    
    def capture_frame(self):
        if not self.autofocus():
            print("Fallo de enfoque, capturando de todos modos...")
            return None
        else:
            return self.picam2.switch_mode_and_capture_array(self.configure_still)

class YoloModel():
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
    
    def preprocess(self, image: np.ndarray):
        height, width = image.shape[:2]
        image_lb, ratio, pad = resize_with_letterbox(image)
        input_tensor = normalize_image(image_lb)

        meta = {
        "ratio": ratio,
        "pad": pad,
        "shape": (height, width)
        }
        return input_tensor, meta
    
    
    def postprocess(self, output0, output1, meta):
        predictions = output0[0]
        prototypes = output1[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        mask_coeffs = predictions[:, 6:]

        boxes, scores, mask_coeffs = filter_detections(boxes, scores, mask_coeffs)
        if len(boxes) == 0:
            return None

        keep = non_max_suppression(boxes, scores, IOU_THRESHOLD)
        boxes, scores, mask_coeffs = boxes[keep], scores[keep], mask_coeffs[keep]

        box, coeff = get_best_detection(boxes, scores, mask_coeffs)

        mask = generate_mask(coeff, prototypes)
        adjusted_box = adjust_box_to_original_scale(box, meta['ratio'], meta['pad'], meta['shape'])
        binary_mask = extract_object_mask(mask, adjusted_box, meta['shape'])
        
        return box, binary_mask

    def __call__(self, image: np.ndarray):
        input_tensor, meta = self.preprocess(image)
        output0, output1 = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(output0, output1, meta)



if __name__ == "__main__":
    import cv2
    camera = Camera()
    frame = camera.capture_frame()
    if frame is not None:
        print("Imagen capturada con éxito")
    else:
        print("No se pudo capturar la imagen")

    frame = cv2.imread('images/prueba/38.jpeg')

    model = YoloModel('models/yolo/YOLO26n-seg-200.onnx')
    result = model(frame)

    if result is not None:
        box, binary_mask = result
        print("Caja:", box)
        print("Coeficientes de máscara:", binary_mask)
        cv2.imwrite("output_mask.png", binary_mask * 255)  # Guardar la máscara binaria para visualización
    else:
        print("No se detectó nada")
