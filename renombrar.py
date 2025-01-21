import os
import pandas as pd
import chardet

# Ruta al directorio del dataset
DATASET_DIR = "C:/datasets/dentiset"
CLASSES = ["cancer", "caries", "gingivitis", "perdidos", "ulceras"]


# Función para detectar la codificación de un archivo
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']


# Función para renombrar imágenes y actualizar el archivo _annotations.csv
def rename_images_and_update_csv(dataset_dir, classes):
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, "train", class_name)
        annotations_path = os.path.join(class_dir, "_annotations.csv")

        # Detectar la codificación del archivo
        encoding = detect_encoding(annotations_path)

        # Leer las anotaciones
        annotations = pd.read_csv(annotations_path, encoding=encoding)

        # Renombrar imágenes
        new_filenames = []
        for idx, row in annotations.iterrows():
            old_filename = row['filename']
            new_filename = f"{class_name}{idx + 1}.jpg"

            old_filepath = os.path.join(class_dir, old_filename)
            new_filepath = os.path.join(class_dir, new_filename)

            # Renombrar archivo de imagen
            if os.path.exists(old_filepath):
                os.rename(old_filepath, new_filepath)

            new_filenames.append(new_filename)

        # Actualizar el archivo _annotations.csv
        annotations['filename'] = new_filenames
        annotations.to_csv(annotations_path, index=False, encoding='utf-8')

        print(f"Clase '{class_name}' procesada. Se renombraron {len(new_filenames)} imágenes.")


# Función principal
if __name__ == "__main__":
    rename_images_and_update_csv(DATASET_DIR, CLASSES)
