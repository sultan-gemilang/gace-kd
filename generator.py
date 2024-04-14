import importlib
import pathlib
import argparse
import time

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import tensorflow_datasets as tfds

# Global Config
AUTOTUNE = tf.data.AUTOTUNE
TRAINING_EPOCHS = 10
VERBOSITY = 1

MODELS_DIR = pathlib.Path(f'{pathlib.Path.cwd()}/generated/oxford_flowers102/')
MODELS_DIR.mkdir(exist_ok=True, parents=True)

def prepare_module(module_name):
    module = importlib.import_module(f'keras.applications.{module_name}')
    return module


def prepare_dataset(dataset, preprocess, batch_size, image_size):
    def resize_image(image, label):
        image = tf.image.resize_with_pad(image, 224, 224)
        image = tf.cast(image, dtype = tf.float32)
        image = preprocess(image)
        return image, label


    (training_set, validation_set, test_set), ds_info = tfds.load(
        name=dataset,
        split=['train[:80%]', 'train[80%:]', 'test'],  # Sesuaikan ini dengan datasetnya
        with_info=True,
        as_supervised=True)

    training_set = training_set.map(map_func=resize_image)
    training_set = training_set \
                    .shuffle(buffer_size=256) \
                    .batch(batch_size) \
                    .prefetch(AUTOTUNE)

    validation_set = validation_set.map(map_func=resize_image)
    validation_set = validation_set.batch(batch_size).prefetch(AUTOTUNE)

    test_set = test_set.map(map_func=resize_image)
    test_set = test_set.batch(batch_size).prefetch(AUTOTUNE)

    num_of_output = ds_info.features['label'].num_classes

    return (training_set, validation_set, test_set, num_of_output)


def build_model(model, base_trainable, image_size, num_of_output):
    base_architecture = model(
        include_top=False,
        input_shape=image_size + (3, ),
        pooling='avg',
    )
    base_architecture.trainable = base_trainable
    outputs = tf.keras.layers.Dense(num_of_output, activation="softmax")(base_architecture.output)

    model = tf.keras.Model(base_architecture.input, outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def generate(model, model_name, training_set, validation_set, test_set, image_size, num_of_output, run_num):
    model = build_model(model=model, 
                        base_trainable=False,
                        image_size=image_size,
                        num_of_output=num_of_output)

    start_time = time.time()
    model.fit(
        training_set,
        epochs=TRAINING_EPOCHS,
        validation_data=validation_set,
        verbose=VERBOSITY,
    )
    end_time = time.time()

    elapsed = round(end_time - start_time, 2)
    print(f"\nFitting took {elapsed} seconds!")


    _, accuracy = model.evaluate(test_set, verbose=VERBOSITY)
    print(f"Model accuracy: {round(accuracy * 100, 2)}")

    # tf.saved_model.save(model, f'{MODELS_DIR}/{model_name}/{run_num}/checkpoint')
    model.save_weights(f'{MODELS_DIR}/{model_name}/{run_num}/checkpoint')
    model.save(f'{MODELS_DIR}/{model_name}/saved-{run_num}')


def main(args):
    module = prepare_module(args.module_name)
    model = getattr(module, args.model_name)
    preprocess_input = getattr(module, 'preprocess_input')
    image_size = (int(args.image_size), int(args.image_size))

    (training_set, validation_set, test_set, num_of_ouput) = prepare_dataset(
        dataset=args.dataset, 
        preprocess=preprocess_input, 
        image_size=image_size,
        batch_size=4)
    
    generate(
        model=model, 
        model_name=args.model_name, 
        training_set=training_set, 
        validation_set=validation_set,
        test_set=test_set, 
        num_of_output=num_of_ouput,
        image_size=image_size,
        run_num=args.run_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset')
    parser.add_argument('--module-name')
    parser.add_argument('--model-name')
    parser.add_argument('--image-size')
    parser.add_argument('--run-num')

    args = parser.parse_args()

    main(args)
