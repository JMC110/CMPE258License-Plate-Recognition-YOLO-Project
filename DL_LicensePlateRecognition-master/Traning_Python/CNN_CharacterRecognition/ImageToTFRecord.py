from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from PIL import Image
import os

# the number of classes of images
classes = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
              'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18,
              'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'P': 24, 'Q': 25, 'R': 26, 'S': 27,
              'T': 28, 'U': 29, 'V': 30, 'W': 31, 'X': 32, 'Y': 33, 'Z': 34}

cwd = "dataset/"
recordPath = "tfrecord/"
bestNum = 1000
num = 0
recordFileNum = 0
recordFileName = ("train.tfrecords-%.3d" % recordFileNum)
writer = tf.io.TFRecordWriter(recordPath + recordFileName)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# to create TFRecords based on sklearn
def convert_to_records(x,y,path):
    print('writing to {}'.format(path))
    with tf.python_io.TFRecordWriter(path) as writer:
        for i in range(x.shape[0]):
            example = tf.train.Example(features=tf.train.Features(
                            feature={'image_raw':_bytes_feature(x[i].tostring()), 'label':_int64_feature(int(y[i]))}))
            writer.write(example.SerializeToString())
            if (i % 5000) == 0:
                print('writing {}th image'.format(i))


# to create TFRecords based on Images
for name, label in classes.items():
    print(name)
    print(label)
    class_path = os.path.join(cwd, name)
    for img_name in os.listdir(class_path):
        num += 1
        if num > bestNum:
            num = 1
            recordFileNum += 1
            writer = tf.io.TFRecordWriter(recordPath + str(recordFileNum))
            print("Creating the %.3d tfrecord file" % recordFileNum)
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path, "r")
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
        writer.write(example.SerializeToString())
writer.close()


