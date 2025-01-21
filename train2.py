import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import pandas as pd
import numpy as np

# Configuración de rutas y parámetros
DATASET_DIR = "C:/datasets/dentiset"
CLASSES = ["cancer", "caries", "gingivitis", "perdidos", "ulceras"]
MODEL_DIR = os.path.join(DATASET_DIR, "modelos")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Función para cargar archivos CSV con diferentes codificaciones
def load_csv_with_encoding(file_path):
    encodings = ["utf-8", "latin1", "iso-8859-1"]
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer el archivo {file_path} con las codificaciones disponibles.")

# Cargar anotaciones
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
label_map = {cls: idx for idx, cls in enumerate(CLASSES)}
for filename in image_annotations:
    image_annotations[filename]['labels'] = [
        label_map[label] for label in image_annotations[filename]['labels']
    ]

# Generador de datos con aumentación
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

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

            X = np.array(X, dtype=np.float32)
            Y_class = np.array(Y_class, dtype=np.int32)
            Y_bbox = np.array(Y_bbox, dtype=np.float32)

            # Aumentación de datos
            for i in range(len(X)):
                X[i] = datagen.random_transform(X[i])

            yield X, {
                "class_output": Y_class,
                "bbox_output": Y_bbox
            }

# Crear un modelo profundo con regularización
def create_deeper_model(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    class_output = layers.Dense(num_classes, activation='softmax', name="class_output")(x)
    bbox_output = layers.Dense(4, activation='sigmoid', name="bbox_output")(x)

    return Model(inputs=inputs, outputs=[class_output, bbox_output])

# Crear y compilar modelo
num_classes = len(label_map)
model = create_deeper_model(num_classes)

def iou_metric(y_true, y_pred):
    x1_true, y1_true, x2_true, y2_true = tf.split(y_true, 4, axis=-1)
    x1_pred, y1_pred, x2_pred, y2_pred = tf.split(y_pred, 4, axis=-1)

    xi1 = tf.maximum(x1_true, x1_pred)
    yi1 = tf.maximum(y1_true, y1_pred)
    xi2 = tf.minimum(x2_true, x2_pred)
    yi2 = tf.minimum(y2_true, y2_pred)

    inter_area = tf.maximum(xi2 - xi1, 0) * tf.maximum(yi2 - yi1, 0)
    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    union_area = true_area + pred_area - inter_area

    return tf.reduce_mean(inter_area / union_area)

model.compile(
    optimizer='adam',
    loss={
        "class_output": "sparse_categorical_crossentropy",
        "bbox_output": "mean_squared_error"
    },
    metrics={
        "class_output": "accuracy",
        "bbox_output": iou_metric
    }
)

# Entrenar modelo
steps_per_epoch = len(image_annotations) // BATCH_SIZE
generator = data_generator(image_annotations, DATASET_DIR, BATCH_SIZE)

history = model.fit(
    generator,
    steps_per_epoch=steps_per_epoch,
    epochs=100
)

# Guardar modelo
model_path = os.path.join(MODEL_DIR, "modelo_dental_mejorado.h5")
model.save(model_path)
print(f"Modelo guardado en: {model_path}")
