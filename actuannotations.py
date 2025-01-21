import os
import pandas as pd


def clean_annotations(dataset_dir, classes):
    """
    Elimina las filas del archivo _annotations.csv cuyo nombre de imagen no existe en el directorio correspondiente.
    :param dataset_dir: Ruta al directorio principal del dataset.
    :param classes: Lista de nombres de carpetas de las clases.
    """
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        annotations_path = os.path.join(class_dir, "_annotations.csv")

        if not os.path.exists(class_dir) or not os.path.exists(annotations_path):
            print(f"Carpeta o archivo _annotations.csv no encontrado para la clase: {class_name}")
            continue

        # Leer el archivo _annotations.csv con codificación 'ISO-8859-1' o 'latin1'
        try:
            annotations = pd.read_csv(annotations_path, encoding='ISO-8859-1')  # o 'latin1'
        except UnicodeDecodeError:
            print(
                f"Error al leer el archivo _annotations.csv en la clase {class_name}. Probando con otra codificación.")
            annotations = pd.read_csv(annotations_path, encoding='latin1')

        # Obtener los archivos en el directorio de la clase
        existing_images = set(os.listdir(class_dir))

        # Filtrar las filas del CSV que tienen imágenes existentes en el directorio
        annotations_cleaned = annotations[annotations['filename'].isin(existing_images)]

        # Guardar el archivo actualizado sin las filas de imágenes no encontradas
        annotations_cleaned.to_csv(annotations_path, index=False)
        print(f"Archivo _annotations.csv limpiado para la clase: {class_name}")


# Configuración
DATASET_DIR = r"C:\datasets\dentiset\train"  # Ruta a tus clases
CLASSES = os.listdir(DATASET_DIR)  # Obtiene automáticamente las carpetas dentro de "train"

clean_annotations(DATASET_DIR, CLASSES)
