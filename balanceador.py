import os
import pandas as pd


def balance_images_and_update_csv(dataset_dir, classes, max_images=650):
    """
    Mantiene solo las primeras 'max_images' imágenes de cada clase y elimina las demás.
    También actualiza el archivo _annotations.csv eliminando las filas correspondientes a las imágenes eliminadas.
    :param dataset_dir: Ruta al directorio principal del dataset.
    :param classes: Lista de nombres de carpetas de las clases.
    :param max_images: Número máximo de imágenes a mantener por clase.
    """
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        annotations_path = os.path.join(class_dir, "_annotations.csv")

        if not os.path.exists(class_dir) or not os.path.exists(annotations_path):
            print(f"Carpeta o archivo _annotations.csv no encontrado para la clase: {class_name}")
            continue

        # Leer el archivo _annotations.csv
        try:
            annotations = pd.read_csv(annotations_path, encoding='ISO-8859-1')  # o 'latin1'
        except UnicodeDecodeError:
            print(
                f"Error al leer el archivo _annotations.csv en la clase {class_name}. Probando con otra codificación.")
            annotations = pd.read_csv(annotations_path, encoding='latin1')

        # Obtener las imágenes actuales
        existing_images = os.listdir(class_dir)

        # Filtrar solo las imágenes que terminan con .jpg o .jpeg
        images = [image for image in existing_images if image.endswith('.jpg') or image.endswith('.jpeg')]

        # Ordenar las imágenes (en caso de que no estén ordenadas)
        images.sort()

        # Seleccionar las primeras 'max_images' imágenes
        images_to_keep = images[:max_images]

        # Eliminar las imágenes que no están en la lista de las primeras 'max_images'
        images_to_delete = set(images) - set(images_to_keep)
        for image in images_to_delete:
            os.remove(os.path.join(class_dir, image))
            print(f"Imagen eliminada: {image}")

        # Filtrar el archivo CSV para mantener solo las filas correspondientes a las imágenes que queremos conservar
        annotations_to_keep = annotations[annotations['filename'].isin(images_to_keep)]

        # Guardar el archivo CSV actualizado con las imágenes restantes
        annotations_to_keep.to_csv(annotations_path, index=False)
        print(f"Archivo _annotations.csv actualizado para la clase: {class_name}")


# Configuración
DATASET_DIR = r"C:\datasets\dentiset\train"  # Ruta a tus clases
CLASSES = os.listdir(DATASET_DIR)  # Obtiene automáticamente las carpetas dentro de "train"

balance_images_and_update_csv(DATASET_DIR, CLASSES, max_images=650)
