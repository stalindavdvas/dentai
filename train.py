import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np

# Ruta al dataset
DATASET_DIR = "C:/datasets/dentiset"
CLASSES = ["cancer", "caries", "gingivitis", "perdidos", "ulceras"]
MODEL_DIR = os.path.join(DATASET_DIR, "modelos")

# Crear carpeta para guardar el modelo
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)  # Reducir resolución
BATCH_SIZE = 16

def load_csv_with_encoding(file_path):
    encodings = ["utf-8", "latin1", "iso-8859-1"]
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer el archivo {file_path} con las codificaciones disponibles.")

def load_annotations(dataset_dir, classes):
    image_data = {}
    for class_name in classes:
        annotations_file = os.path.join(dataset_dir, "train", class_name, "_annotations.csv")
        if not os.path.exists(annotations_file):
            print(f"Archivo de anotaciones no encontrado: {annotations_file}")
            continue

        annotations = load_csv_with_encoding(annotations_file)
        for _, row in annotations.iterrows():
            filename = os.path.join("train", class_name, row['filename'])
            label = class_name
            box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            width = row['width']
            height = row['height']

            normalized_box = normalize_boxes([box], width, height)[0]

            if filename not in image_data:
                image_data[filename] = {'boxes': [], 'labels': []}
            image_data[filename]['boxes'].append(normalized_box)
            image_data[filename]['labels'].append(label)

    return image_data

# Normalizar cajas delimitadoras
def normalize_boxes(boxes, width, height):
    return np.array([
        [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
        for box in boxes
    ], dtype=np.float32)

# Procesar anotaciones
image_annotations = load_annotations(DATASET_DIR, CLASSES)

# Mapear clases a índices
label_map = {"cancer": 0, "caries": 1, "gingivitis": 2, "perdidos": 3, "ulceras": 4}
for filename in image_annotations:
    image_annotations[filename]['labels'] = [
        label_map[label] for label in image_annotations[filename]['labels']
    ]

# Generador de datos

def data_generator(image_annotations, dataset_dir, batch_size):
    filenames = list(image_annotations.keys())
    num_samples = len(filenames)

    while True:
        for offset in range(0, num_samples, batch_size):
            batch_filenames = filenames[offset:offset + batch_size]

            X, Y_class, Y_bbox = [], [], []

            for filename in batch_filenames:
                img_path = os.path.join(dataset_dir, filename)

                if not os.path.exists(img_path):
                    print(f"Advertencia: La imagen {img_path} no se encuentra. Será omitida.")
                    continue

                img = load_img(img_path, target_size=IMG_SIZE)
                img_array = img_to_array(img) / 255.0

                boxes = image_annotations[filename]['boxes']
                labels = image_annotations[filename]['labels']

                X.append(img_array)
                Y_class.append(labels[0])
                Y_bbox.append(boxes[0])

            yield np.array(X, dtype=np.float32), {
                "class_output": np.array(Y_class, dtype=np.int32),
                "bbox_output": np.array(Y_bbox, dtype=np.float32)
            }

# Crear modelo
def create_model(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)

    class_output = layers.Dense(num_classes, activation='softmax', name="class_output")(x)
    bbox_output = layers.Dense(4, activation='sigmoid', name="bbox_output")(x)

    model = Model(inputs=inputs, outputs=[class_output, bbox_output])
    return model

num_classes = len(label_map)
model = create_model(num_classes)

# Compilar modelo
model.compile(
    optimizer='adam',
    loss={
        "class_output": "sparse_categorical_crossentropy",
        "bbox_output": "mean_squared_error"
    },
    metrics={
        "class_output": "accuracy",
        "bbox_output": "mse"
    }
)

# Parámetros de entrenamiento
steps_per_epoch = len(image_annotations) // BATCH_SIZE
generator = data_generator(image_annotations, DATASET_DIR, BATCH_SIZE)

# Entrenar modelo
history = model.fit(
    generator,
    steps_per_epoch=steps_per_epoch,
    epochs=1
)

# Guardar modelo
model_path = os.path.join(MODEL_DIR, "modelo_dental.h5")
model.save(model_path)
print(f"Modelo guardado en: {model_path}")
