import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, weights_file, config_file, classes_file, filtered_class):
        self.net = cv2.dnn.readNet(weights_file, config_file)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255)
        self.filtered_class = [int(i) for i in filtered_class.split(",")]
        
        with open(classes_file, "r") as file_object:
            self.classes = [line.strip() for line in file_object]

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect_objects(self, img, nms_threshold=0.3, debug= False):
        class_detections = {}
        # img = cv2.imread(image_file)
        
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=nms_threshold)

        for (class_id, score, box) in zip(class_ids, scores, boxes):
            if int(class_id) in self.filtered_class:
                x, y, w, h = box
                class_id = class_id
                class_name = self.classes[class_id]
                color = self.colors[class_id]

                if class_id not in class_detections:
                    class_detections[class_id] = []

                class_detections[class_id].append({
                    'class_name': class_name,
                    'score': score
                    # 'box': (x, y, w, h),
                    # 'color': color
                })

                if debug:
                    cv2.putText(img, "{} {}".format(class_name, score), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                
        if debug and bool(class_detections):
            cv2.imshow("Img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return class_detections

if __name__ == "__main__":
    detector = ObjectDetector("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg", "dnn_model/classes.txt")
    results = detector.detect_objects("prueba2.jpg")
    print(results)
