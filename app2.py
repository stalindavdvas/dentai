from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import custom_object_scope
from werkzeug.utils import secure_filename

# Inicializar la aplicación Flask
app = Flask(__name__)

# Rutas de almacenamiento
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Mapeo de clases
label_map = {0: "cancer", 1: "caries", 2: "gingivitis", 3: "perdidos", 4: "ulceras"}

# Definir la métrica personalizada
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

# Cargar el modelo con la métrica personalizada
MODEL_PATH = "modelos/modelo_dental.h5"
with custom_object_scope({'iou_metric': iou_metric}):
    model = load_model(MODEL_PATH)

# Función para preprocesar la imagen
def preprocess_image(image_path):
    original_img = cv2.imread(image_path)
    img_resized = cv2.resize(original_img, (512, 512))  # Ajustar a 512x512
    img_array = img_to_array(img_resized) / 255.0  # Normalizar entre 0 y 1
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch
    return original_img, img_array

# Función para realizar la predicción
def predict(image_path):
    original_img, img_array = preprocess_image(image_path)

    # Realizar la predicción
    class_output, bbox_output = model.predict(img_array)

    # Obtener la clase predicha
    predicted_class = np.argmax(class_output, axis=1)[0]
    class_name = label_map[predicted_class]
    confidence = class_output[0][predicted_class]

    # Obtener las coordenadas de la caja delimitadora
    bbox = bbox_output[0]
    height, width, _ = original_img.shape
    x1, y1, x2, y2 = (bbox * [width, height, width, height]).astype(int)

    # Dibujar la caja delimitadora sobre la imagen original
    color = (0, 255, 0)
    thickness = 2
    cv2.rectangle(original_img, (x1, y1), (x2, y2), color, thickness)

    # Agregar el texto (nombre de la enfermedad y confianza)
    text = f"{class_name}: {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 0, 255)
    thickness = 2
    cv2.putText(original_img, text, (x1, y1 - 10), font, font_scale, font_color, thickness)

    # Guardar las imágenes en las rutas estáticas
    result_image_path = os.path.join(RESULTS_FOLDER, 'result.jpg')
    cv2.imwrite(result_image_path, original_img)

    # Retornar las rutas de las imágenes
    return result_image_path, class_name, confidence

# Ruta para servir el HTML (página de inicio)
@app.route('/')
def index():
    return render_template('inicio1.html')

# Ruta para manejar la carga de la imagen
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Guardar la imagen original
    filename = secure_filename(file.filename)
    original_image_path = os.path.join(UPLOAD_FOLDER, 'original.jpg')

    # Eliminar la imagen anterior si existe
    if os.path.exists(original_image_path):
        os.remove(original_image_path)

    # Guardar la nueva imagen
    file.save(original_image_path)

    # Procesar la imagen y obtener los resultados
    result_image_path, disease, confidence = predict(original_image_path)

    # Devolver los resultados
    return jsonify({
        'original': original_image_path,
        'result': result_image_path,
        'detections': [{
            'disease': disease,
            'confidence': f"{confidence * 100:.1f}%"
        }]
    })

if __name__ == '__main__':
    app.run(debug=True)
