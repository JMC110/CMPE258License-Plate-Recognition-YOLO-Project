import cv2 as cv
import tensorflow as tf
import numpy as np


class Predictor:

    # Darknet and CNN Parameters
    confidence_threshold = 0.1  # Confidence threshold
    nms_threshold = 0.6  # Non-maximum suppression threshold

    yolo_net_width = 416
    yolo_net_height = 416

    # Load all models and configs
    plates_yolo_config = "dl_prediction/config/yolov3_plates.cfg"
    plates_yolo_weights = "dl_prediction/models/yolov3_plates_final.weights"
    plates_classes = ['Plate']

    chars_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                     'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    chars_yolo_config = "dl_prediction/config/yolov3_chars.cfg"
    chars_yolo_weights = "dl_prediction/models/yolov3_chars_final.weights"

    def __init__(self):
        self.plates_yolo_net = self.get_yolo_net(self.plates_yolo_config, self.plates_yolo_weights)
        self.chars_yolo_net = self.get_yolo_net(self.chars_yolo_config, self.chars_yolo_weights)
        self.cnn_chars_model = tf.keras.models.load_model('dl_prediction/models/cnn_chars_recognition.h5')

    @staticmethod
    def get_yolo_net(config, weights):
        yolo_net = cv.dnn.readNetFromDarknet(config, weights)
        yolo_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        yolo_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        return yolo_net

    @staticmethod
    def process_license_plate(license_plate):
        gray = cv.cvtColor(license_plate, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, 0)

        contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        areas = [cv.contourArea(c) for c in contours]

        if len(areas) != 0:
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(license_plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
            processed_license_plate = license_plate[y: y + h, x: x + w]
        else:
            processed_license_plate = license_plate

        return processed_license_plate

    @staticmethod
    def draw_pred(frame, name, conf, left, top, right, bottom, color=(0, 255, 0)):
        cv.rectangle(frame, (left, top), (right, bottom), color, 3)
        # label = '{}:{}'.format(name, '%.2f' % conf)
        label = name
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - round(1.5 * label_size[1])),
                     (left + round(1.5 * label_size[0]), top + base_line), (0, 0, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    def predict_boxes(self, frame, yolo_outputs, is_license_plate=True):
        classes = []
        confidences = []
        boxes = []

        max_confidence = 0.0
        for output in yolo_outputs:
            for prediction in output:
                scores = prediction[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                max_confidence = max(confidence, max_confidence)
                if confidence > self.confidence_threshold:
                    center_x = int(prediction[0] * frame.shape[1])
                    center_y = int(prediction[1] * frame.shape[0])

                    width = int(prediction[2] * frame.shape[1])
                    height = int(prediction[3] * frame.shape[0])
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)

                    classes.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        positions = []
        chars = []

        for index in indices:
            index = index[0]
            box = boxes[index]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            positions.append(left)

            if is_license_plate and max_confidence == confidences[index]:
                # Draw prediction rectangle for License Plate

                license_plate = frame[top: top + height, left: left + width]
                cv.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 3)
                # cv.imshow('License Plate', license_plate)
                # cv.imwrite('./test_bha.jpg', license_plate.astype(np.uint8))

                # Process Licence plate to cover to Gray and to enhance contours
                processed_license_plate = self.process_license_plate(license_plate)
                # cv.imshow('Processed License Plate', license_plate)
                # cv.imwrite('./test_bha2.jpg', processed_license_plate.astype(np.uint8))

                self.draw_pred(frame, self.plates_classes[0], confidences[index], left, top, left + width, top + height)
                return "", processed_license_plate
            else:
                char = self.chars_classes[classes[index]]
                chars.append(char)
                self.draw_pred(frame, char, confidences[index], left, top, left + width, top + height, color=(90, 0, 255))

        sorted_chars = [x for _, x in sorted(zip(positions, chars))]
        return "".join(sorted_chars), frame

    def cnn_char_recognition(self, image):
        gray_char = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray_char = cv.resize(gray_char, (75, 100))
        image = gray_char.reshape((1, 100, 75, 1))
        image = image / 255.0
        predictions = self.cnn_chars_model.predict(image)
        max_confidence_index = np.argmax(predictions)
        return self.chars_classes[max_confidence_index]

    def canny(self, image, sigma=0.33):
        lower = int(max(0, (1.0 - sigma) * np.median(image)))
        upper = int(min(255, (1.0 + sigma) * np.median(image)))
        edges = cv.Canny(image, lower, upper)
        return edges

    def cnn_recognize_plate(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        thresh_inv = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 39, 1)
        edges = self.canny(thresh_inv)

        contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        sorted_ctrs = sorted(contours, key=lambda x: cv.boundingRect(x)[0])
        area = frame.shape[0] * frame.shape[1]

        chars = []
        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv.boundingRect(ctr)
            roi_area = w * h
            non_max_sup = roi_area / area

            if (non_max_sup >= 0.015) and (non_max_sup < 0.09):
                if (h > 1.2 * w) and (3 * w >= h):
                    char = frame[y:y + h, x:x + w]
                    chars.append(self.cnn_char_recognition(char))
                    cv.rectangle(frame, (x, y), (x + w, y + h), (90, 0, 255), 2)
        licensePlate = "".join(chars)
        return licensePlate

    @staticmethod
    def resize_license_plate(license_plate):
        scale_percent = 300  # percent of original size
        width = int(license_plate.shape[1] * scale_percent / 100)
        height = int(license_plate.shape[0] * scale_percent / 100)
        return cv.resize(license_plate, (width, height), interpolation=cv.INTER_AREA)

    def get_image_blob(self, image):
        return cv.dnn.blobFromImage(image, 1 / 255, (self.yolo_net_width, self.yolo_net_height), [0, 0, 0], 1,
                                    crop=False)

    def predict(self, input_path, output_car_path, output_license_path_original, output_license_path, video_path=None, is_cnn=False, is_image=True):
        vc = cv.VideoCapture(input_path)

        FPS = 2
        # Limit number of frames per sec
        vc.set(cv.CAP_PROP_FPS, FPS)

        if not is_image:
            vid_writer = cv.VideoWriter(video_path, cv.VideoWriter_fourcc('H','2','6','4'), 30, (
            round(vc.get(cv.CAP_PROP_FRAME_WIDTH)), round(vc.get(cv.CAP_PROP_FRAME_HEIGHT))))

        # Read Frames from the file if video, else read first frame from image
        while cv.waitKey(1) < 0:
            exists, frame = vc.read()
            if not exists:
                cv.waitKey(2000)
                print("End of frames")
                vid_writer.release()
                break

            # DETECT LICENSE PLATE

            car_image_blob = self.get_image_blob(frame)

            # Feed the input image to the Yolo Network
            self.plates_yolo_net.setInput(car_image_blob)

            # Get All Unconnected Yolo layers
            plates_yolo_layers = [self.plates_yolo_net.getLayerNames()[i[0] - 1] for i in
                                  self.plates_yolo_net.getUnconnectedOutLayers()]

            # Forward pass the input to yolov3 net and get outputs
            plates_output = self.plates_yolo_net.forward(plates_yolo_layers)

            # Remove the bounding boxes with low confidence and draw box for license plate
            license_num, processed_license_plate = self.predict_boxes(frame, plates_output)
            if is_image:
                cv.imwrite(output_license_path_original, processed_license_plate.astype(np.uint8))

            if not is_cnn:
                # IDENTIFY LICENSE PLATE NUMBER USING YOLOV3

                # Resize the license plate so we can feed it to the second trained yolo network
                resized_license_plate = self.resize_license_plate(processed_license_plate)

                license_plate_image_blob = self.get_image_blob(resized_license_plate)
                # license_plate_image_blob = np.reshape(license_plate_image_blob, (1, 3, yolo_net_width,
                # yolo_net_height))

                # Feed the input image to the Yolo Network
                self.chars_yolo_net.setInput(license_plate_image_blob)

                # Get All Unconnected Yolo layers
                chars_yolo_layers = [self.chars_yolo_net.getLayerNames()[i[0] - 1] for i in
                                     self.chars_yolo_net.getUnconnectedOutLayers()]

                # Forward pass the input to yolov3 net and get outputs
                chars_output = self.chars_yolo_net.forward(chars_yolo_layers)

                license_number, processed_license_plate = self.predict_boxes(processed_license_plate, chars_output, is_license_plate=False)
                print(license_number)

            elif is_cnn:
                # IDENTIFY LICENSE PLATE NUMBER USING CNN
                license_number = self.cnn_recognize_plate(processed_license_plate)

            if is_image:
                cv.imwrite(output_license_path, processed_license_plate.astype(np.uint8))
                cv.imwrite(output_car_path, frame.astype(np.uint8))
                return license_number
            else:
                vid_writer.write(frame.astype(np.uint8))


if __name__ == "__main__":
    Predictor()

# license = Predictor().predict('/Users/moni/projects/LicensePlateDetector/m657_1.jpg', False)
# print(license)
