# Code inspiration from Darknet scripts

import glob
import os
import xml.etree.ElementTree as et
from sklearn.model_selection import train_test_split

classes = ["plate"]

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(input_file, output_file):
    tree = et.parse(input_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    if (w ==0 or h ==0):
        raise Exception('width and height of the annotation cannot be 0. {}', input_file)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        output_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def generate_darknet_config_files(images_input_dir, annotations_input_dir, labels_output_dir, cfg_output_dir, rel_path_prefix):
    image_file_paths = []
    for image_file_path in glob.glob(images_input_dir + '/*.jpg'):
        image_file_paths.append(rel_path_prefix + image_file_path)
        filename = image_file_path.split('/')[-1].split('.')[0]
        annotation_file_path = '{}/{}.xml'.format(annotations_input_dir, filename)
        if not os.path.exists(annotation_file_path):
            raise Exception('Annotation not found for an image {}'.format(image_file_path))
        label_file_path = '{}/{}.txt'.format(labels_output_dir, filename)

        if not os.path.exists(labels_output_dir):
            os.mkdir(labels_output_dir)

        in_file = open(annotation_file_path)
        out_file = open(label_file_path, 'x')
        convert_annotation(in_file, out_file)
        in_file.close()
        out_file.close()

    X_train, X_test = train_test_split(image_file_paths, test_size=0.05, shuffle=True)
    train_file = open(cfg_output_dir + '/train.txt', 'x')
    train_file.writelines("%s\n" % path for path in X_train)
    train_file.close()

    test_file = open(cfg_output_dir + '/test.txt', 'x')
    test_file.writelines("%s\n" % path for path in X_test)
    test_file.close()


images_input_dir = 'dataset/belgium/images'
annotations_input_dir = 'dataset/belgium/annotations'
labels_output_dir = 'dataset/belgium/labels'
cfg_output_dir = 'dataset/belgium'

generate_darknet_config_files(images_input_dir, annotations_input_dir, labels_output_dir, cfg_output_dir, "../Yolo_LicensePlateDetection/")