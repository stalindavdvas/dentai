import os
import pandas as pd


def rename_images_and_update_csv(dataset_dir, classes):
    """
    Renombra las imágenes de cada clase y actualiza el archivo _annotations.csv con los nuevos nombres.
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

        # Obtener las imágenes actuales
        existing_images = os.listdir(class_dir)

        # Filtrar solo las imágenes
        images = [image for image in existing_images if image.endswith('.jpg') or image.endswith('.jpeg')]

        # Crear un nuevo nombre de archivo para cada imagen
        image_mapping = {}
        for idx, image in enumerate(images, start=1):
            new_name = f"{class_name}{idx}.jpg"
            os.rename(os.path.join(class_dir, image), os.path.join(class_dir, new_name))
            image_mapping[image] = new_name

        # Actualizar los nombres de las imágenes en el DataFrame
        annotations['filename'] = annotations['filename'].apply(lambda x: image_mapping.get(x, x))

        # Guardar el archivo actualizado con los nuevos nombres
        annotations.to_csv(annotations_path, index=False)
        print(f"Archivo _annotations.csv actualizado para la clase: {class_name}")


# Configuración
DATASET_DIR = r"C:\datasets\dentiset\train"  # Ruta a tus clases
CLASSES = os.listdir(DATASET_DIR)  # Obtiene automáticamente las carpetas dentro de "train"

rename_images_and_update_csv(DATASET_DIR, CLASSES)
