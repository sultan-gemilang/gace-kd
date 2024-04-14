import os
from datetime import datetime 
import psutil
import threading
import argparse
import importlib
from utils.distiller import Distiller
import pathlib
import time
from utils.monitor import CPU, Memory

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import tensorflow_datasets as tfds


# ====================== GENERAL CONFIG =======================
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)
EPOCHS = 5
VERBOSITY = 1


# ====================== PREPROCESSOR =========================
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


# ====================== SET TEACHER & STUDENT ================
def get_teacher_model(model_name, run_num):
    baseline_path = pathlib.Path(f'{pathlib.Path.cwd()}/generated/oxford_flowers102/{model_name}/saved-{run_num}')
    print(f'{baseline_path} as teacher')
    return tf.saved_model.load(baseline_path)


def set_student_model(student_model: tf.keras.Model, num_of_output):
    base_architecture = student_model(
        include_top=False,
        input_shape=IMAGE_SIZE + (3, ),
        pooling='avg',
    )
    base_architecture.trainable = False

    outputs = tf.keras.layers.Dense(num_of_output, activation="softmax")(base_architecture.output)
    model = tf.keras.Model(base_architecture.input, outputs)

    return model


# ====================== MAIN =================================
def test_distilled_student(student_model_path, test_set):
    student: tf.keras.Model = tf.keras.models.load_model(student_model_path)
    student.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    student.evaluate(test_set, verbose=2)


def generate(teacher_model_name, student_model, student_model_name, training_set, validation_set, test_set, num_of_output, run_num):
    teacher = get_teacher_model(teacher_model_name, run_num)

    student = set_student_model(student_model, num_of_output)
    print(f'Set {student_model_name} as Student!')

    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        metrics=['accuracy'],
        alpha=0.1,
        temperature=10,
    )

    # Distill teacher to student
    start_time = time.time()
    distiller.fit(
        training_set, 
        validation_data=validation_set, 
        verbose=VERBOSITY,  
        epochs=EPOCHS)
    end_time = time.time()
    elapsed = round(end_time - start_time, 2)
    print(f'KD with {EPOCHS} epochs took {elapsed}s!')

    student_path = f'{pathlib.Path.cwd()}/generated/kd/{teacher_model_name}_{student_model_name}'
    
    distiller.student.save(student_path)
    test_distilled_student(student_path, test_set)


def main(args):
    os.system("echo 'CloudLab12#$%' | sudo -S sync; sudo -S su -c 'echo 3 > /proc/sys/vm/drop_caches'")

    module = prepare_module(args.module_name)
    student_model = getattr(module, args.student_model)
    preprocess_input = getattr(module, 'preprocess_input')

    (training_set, validation_set, test_set, num_of_ouput) = prepare_dataset(
        dataset='oxford_flowers102', # Bisa ganti ke cifar100 atau yg lain, cek di: https://www.tensorflow.org/datasets/catalog/overview
        preprocess=preprocess_input, 
        batch_size=4) # Ini jg disesuaikan sama dataset, kalau dataset besar, batch boleh digedein

    for i in range(1, 2):
        pid = os.getpid()
        begin = datetime.now()

        generate(
            teacher_model_name=args.teacher,
            student_model=student_model,
            student_model_name=args.student_model,
            training_set=training_set, 
            validation_set=validation_set,
            test_set=test_set, 
            num_of_output=num_of_ouput,
            run_num=args.run_num)

        # Create event to signal task completion
        task_completed = threading.Event()

        # Start monitoring thread
        cpu_monitor = CPU(pid, task_completed)
        mem_monitor = Memory(pid, task_completed)
        cpu_monitor.start()
        mem_monitor.start()
        # _jstat_start()

        # Wait for the subprocess to complete
        process = psutil.Process(pid)
        process.wait()
        end = datetime.now()

        elapsed = (end - begin).total_seconds()

        # Signal task completion
        task_completed.set()

        # Wait for the monitoring thread to finish
        cpu_monitor.join()
        mem_monitor.join()

        cpu_data = cpu_monitor.cpu_usage_data
        mem_data = mem_monitor.mem_usage_data

        print(f'CPU Utils: {cpu_data}')
        print(f'Mem Utils: {mem_data}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--module-name') # preprocessing buat student
    parser.add_argument('--teacher') # nama teacher model
    parser.add_argument('--student-model') # nama student model
    parser.add_argument('--run-num') # run number: 1, 2, 3... etc
    
    args = parser.parse_args()
    main(args)
