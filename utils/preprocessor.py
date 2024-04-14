def prepare_module(module_name):
    module = importlib.import_module(f'keras.applications.{module_name}')
    return module


def prepare_dataset(dataset, preprocess, batch_size):
    def resize_image(image, label):
        image = tf.image.resize(image, size=IMAGE_SIZE)
        image = tf.cast(image, dtype = tf.float32)
        image = preprocess(image)
        return image, label


    (training_set, validation_set, test_set), ds_info = tfds.load(
        name=dataset,
        split=['train', 'validation', 'test'],
        with_info=True,
        as_supervised=True)

    training_set = training_set.map(map_func=resize_image)
    training_set = training_set.shuffle(256) \
                    .batch(batch_size) \
                    .prefetch(AUTOTUNE)

    validation_set = validation_set.map(map_func=resize_image, num_parallel_calls=AUTOTUNE)
    validation_set = validation_set.batch(batch_size=batch_size).prefetch(AUTOTUNE)

    test_set = test_set.map(map_func=resize_image)
    test_set = test_set.batch(batch_size).prefetch(AUTOTUNE)

    num_of_output = ds_info.features['label'].num_classes

    return (training_set, validation_set, test_set, num_of_output)