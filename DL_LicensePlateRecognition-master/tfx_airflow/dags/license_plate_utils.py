import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

_IMAGE_KEY = 'img_raw'
_LABEL_KEY = 'label'

channel = 1
height = 100
width = 75


def _transformed_name(key):
    return key + '_xf'

def _image_parser(image_str):
    image = tf.image.decode_image(image_str, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, (28400, 100, 75, 1))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def _label_parser(label_id):
    label = tf.one_hot(label_id, 35, dtype=tf.int64)
    return label


def preprocessing_fn(inputs):
    outputs = {_transformed_name(_IMAGE_KEY): tf.compat.v2.map_fn(_image_parser, tf.squeeze(inputs[_IMAGE_KEY], axis=1),
                                                                  dtype=tf.float32),
               _transformed_name(_LABEL_KEY): tf.compat.v2.map_fn(_label_parser, tf.squeeze(inputs[_LABEL_KEY], axis=1),
                                                                  dtype=tf.int64)
               }
    return outputs

# CNN model to predict characters
def _model_builder():
    cnn_model = models.Sequential()
    cnn_model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (100, 75, 1)))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Dense(128, activation = 'relu'))
    cnn_model.add(layers.Flatten())
    # 0 - 9 and A-Z => 10 + 25 = 35  -- ignoring O in alphabets.
    cnn_model.add(layers.Dense(35, activation = 'softmax'))
    opt = Adam(lr=0.001)
    cnn_model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    absl.logging.info(cnn_model.summary())
    return cnn_model

def _serving_input_receiver_fn(tf_transform_output):
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_feature_spec.pop(_LABEL_KEY)
    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(raw_feature_spec,
                                                                               default_batch_size=None)
    serving_input_receiver = raw_input_fn()
    transformed_features = tf_transform_output.transform_raw_features(serving_input_receiver.features)
    transformed_features.pop(_transformed_name(_LABEL_KEY))
    return tf.estimator.export.ServingInputReceiver(transformed_features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(tf_transform_output):
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(raw_feature_spec,
                                                                               default_batch_size=None)
    serving_input_receiver = raw_input_fn()
    transformed_features = tf_transform_output.transform_raw_features(serving_input_receiver.features)
    transformed_labels = transformed_features.pop(_transformed_name(_LABEL_KEY))
    return tfma.export.EvalInputReceiver(features=transformed_features, labels=transformed_labels,
                                         receiver_tensors=serving_input_receiver.receiver_tensors)


def _input_fn(filenames, tf_transform_output, batch_size):
    transformed_feature_spec = (tf_transform_output.transformed_feature_spec().copy())
    dataset = tf.data.experimental.make_batched_features_dataset(filenames, batch_size, transformed_feature_spec)
    return dataset.map(lambda features: (features, features.pop(_transformed_name(_LABEL_KEY))))


def trainer_fn(trainer_fn_args, schema):  # pylint: disable=unused-argument
    train_batch_size = 32
    eval_batch_size = 32
    tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)
    train_input_fn = lambda: _input_fn(trainer_fn_args.train_files, tf_transform_output, batch_size=train_batch_size)
    eval_input_fn = lambda: _input_fn(trainer_fn_args.eval_files, tf_transform_output, batch_size=eval_batch_size)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=trainer_fn_args.train_steps)
    serving_receiver_fn = lambda: _serving_input_receiver_fn(tf_transform_output)
    exporter = tf.estimator.FinalExporter('license_plate', serving_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=trainer_fn_args.eval_steps, exporters=[exporter],
                                      name='license_plate')
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=999, keep_checkpoint_max=1)
    run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)
    estimator = tf.keras.estimator.model_to_estimator(keras_model=_model_builder(), config=run_config)
    eval_receiver_fn = lambda: _eval_input_receiver_fn(tf_transform_output)

    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': eval_receiver_fn
    }