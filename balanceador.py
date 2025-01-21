import os
import pandas as pd
import random
import shutil
import chardet

# Ruta al directorio del dataset
DATASET_DIR = "C:/datasets/dentiset"
CLASSES = ["cancer", "caries", "gingivitis", "perdidos", "ulceras"]
MAX_IMAGES = 600  # Número máximo de imágenes por clase


# Función para detectar la codificación de un archivo
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']


# Función para ajustar las clases a un número fijo de imágenes
def adjust_classes_to_limit(dataset_dir, classes, max_images):
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, "train", class_name)
        annotations_path = os.path.join(class_dir, "_annotations.csv")

        # Detectar la codificación del archivo
        encoding = detect_encoding(annotations_path)

        # Leer las anotaciones
        annotations = pd.read_csv(annotations_path, encoding=encoding)

        # Si las imágenes son más que el límite, reducir
        if len(annotations) > max_images:
            print(f"Clase '{class_name}' tiene {len(annotations)} imágenes, reduciendo a {max_images}.")

            # Seleccionar aleatoriamente las primeras `max_images`
            annotations = annotations.sample(n=max_images, random_state=42).reset_index(drop=True)

            # Eliminar imágenes que no están en el conjunto reducido
            images_to_keep = set(annotations['filename'])
            all_images = set(os.listdir(class_dir))
            images_to_delete = all_images - images_to_keep - {"_annotations.csv"}

            for image_name in images_to_delete:
                image_path = os.path.join(class_dir, image_name)
                if os.path.exists(image_path):
                    os.remove(image_path)

            # Guardar el archivo CSV actualizado
            annotations.to_csv(annotations_path, index=False, encoding='utf-8')
            print(f"Clase '{class_name}' ajustada a {len(annotations)} imágenes.")

        # Si ya tiene menos imágenes que el límite, no hacemos nada
        else:
            print(f"Clase '{class_name}' ya tiene {len(annotations)} imágenes, no se necesita ajuste.")


# Función principal
if __name__ == "__main__":
    adjust_classes_to_limit(DATASET_DIR, CLASSES, MAX_IMAGES)
