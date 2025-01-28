from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import os
import cv2
import numpy as np
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import custom_object_scope
from werkzeug.utils import secure_filename
from datetime import datetime
import tensorflow as tf
from pathlib import PosixPath, WindowsPath
import pathlib

pathlib.PosixPath = pathlib.WindowsPath
app = Flask(__name__)
socketio = SocketIO(app)

# Configurar directorios fijos
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Cargar modelo YOLO
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='modelos/best.pt')

# Mapeo de clases para el modelo personalizado
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

# Cargar el modelo personalizado con la métrica personalizada
MODEL_PATH = "modelos/modelo_dental.h5"
with custom_object_scope({'iou_metric': iou_metric}):
    custom_model = load_model(MODEL_PATH)

def get_treatment_recommendation(disease, confidence):
    """Retorna recomendaciones basadas en la enfermedad detectada y su confianza"""
    recommendations = {
        'caries': {
            'leve': "Se detectó caries con baja severidad. Recomendaciones:\n"
                    "- Limpieza dental profesional\n"
                    "- Mejorar higiene bucal diaria\n"
                    "- Usar pasta dental con flúor",
            'moderada': "Se detectó caries moderada. Recomendaciones:\n"
                        "- Tratamiento de empaste dental\n"
                        "- Evaluación de la profundidad de la caries\n"
                        "- Posible tratamiento de conducto",
            'severa': "Se detectó caries severa. Recomendaciones:\n"
                      "- Tratamiento urgente de conducto\n"
                      "- Posible extracción dental\n"
                      "- Evaluación completa de la estructura dental"
        },
        'gingivitis': {
            'leve': "Se detectó gingivitis leve. Recomendaciones:\n"
                    "- Mejorar rutina de higiene bucal\n"
                    "- Uso de enjuague bucal antiséptico\n"
                    "- Limpieza dental profesional",
            'moderada': "Se detectó gingivitis moderada. Recomendaciones:\n"
                        "- Tratamiento periodontal básico\n"
                        "- Uso de antibióticos tópicos\n"
                        "- Seguimiento cercano",
            'severa': "Se detectó gingivitis severa. Recomendaciones:\n"
                      "- Tratamiento periodontal profundo\n"
                      "- Posible cirugía gingival\n"
                      "- Tratamiento antibiótico"
        }
        # Agregar más enfermedades según tu modelo
    }

    # Determinar severidad basada en la confianza
    if confidence < 0.5:
        severity = 'leve'
    elif confidence < 0.8:
        severity = 'moderada'
    else:
        severity = 'severa'

    return recommendations.get(disease, {}).get(severity, "No hay recomendaciones específicas disponibles.")

def process_image_with_yolo(image_path):
    # Cargar imagen original
    original_img = cv2.imread(image_path)

    # Realizar detección
    results = yolo_model(image_path)

    # Obtener detecciones
    detections = results.xyxy[0].cpu().numpy()  # xyxy format: x1, y1, x2, y2, confidence, class

    # Dibujar solo las cajas delimitadoras en la imagen original
    detected_diseases = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        # Convertir a enteros
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Obtener etiqueta y confianza
        label = yolo_model.names[int(cls)]
        confidence = float(conf)

        # Dibujar caja delimitadora
        color = (0, 255, 0)  # Color verde para las cajas
        thickness = 2
        cv2.rectangle(original_img, (x1, y1), (x2, y2), color, thickness)

        # Dibujar etiqueta
        label_text = f'{label} {confidence:.2f}'
        cv2.putText(original_img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        # Guardar información para recomendaciones
        detected_diseases.append((label, confidence))

    # Guardar imagen resultante
    result_image_path = os.path.join(RESULTS_FOLDER, 'result_yolo.jpg')
    cv2.imwrite(result_image_path, original_img)

    return result_image_path, detected_diseases

def process_image_with_custom_model(image_path):
    original_img = cv2.imread(image_path)
    img_resized = cv2.resize(original_img, (512, 512))  # Ajustar a 512x512
    img_array = img_to_array(img_resized) / 255.0  # Normalizar entre 0 y 1
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch

    # Realizar la predicción
    class_output, bbox_output = custom_model.predict(img_array)

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
    result_image_path = os.path.join(RESULTS_FOLDER, 'result_custom.jpg')
    cv2.imwrite(result_image_path, original_img)

    return result_image_path, class_name, confidence

@app.route('/')
def home():
    return render_template('inicio2.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'})

    if file:
        try:
            # Limpiar imágenes anteriores
            original_image_path = os.path.join(UPLOAD_FOLDER, 'original.jpg')
            if os.path.exists(original_image_path):
                os.remove(original_image_path)

            # Guardar nueva imagen original
            file.save(original_image_path)

            # Procesar imagen con ambos modelos
            result_yolo_path, detected_diseases_yolo = process_image_with_yolo(original_image_path)
            result_custom_path, disease_custom, confidence_custom = process_image_with_custom_model(original_image_path)

            # Preparar resultados de detección
            detection_results_yolo = []
            if detected_diseases_yolo:
                for disease, confidence in detected_diseases_yolo:
                    detection_results_yolo.append({
                        'disease': disease,
                        'confidence': f"{confidence * 100:.1f}%"
                    })
                    recommendation = get_treatment_recommendation(disease.lower(), confidence)
                    timestamp = datetime.now().strftime('%H:%M')
                    socketio.emit('message', {
                        'msg': f"Detección (YOLO): {disease} (Confianza: {confidence:.2f})\n\n{recommendation}",
                        'timestamp': timestamp
                    })

            detection_results_custom = [{
                'disease': disease_custom,
                'confidence': f"{confidence_custom * 100:.1f}%"
            }]
            recommendation_custom = get_treatment_recommendation(disease_custom.lower(), confidence_custom)
            timestamp = datetime.now().strftime('%H:%M')
            socketio.emit('message', {
                'msg': f"Detección (Modelo Personalizado): {disease_custom} (Confianza: {confidence_custom:.2f})\n\n{recommendation_custom}",
                'timestamp': timestamp
            })

            return jsonify({
                'original': '/static/uploads/original.jpg',
                'result_yolo': '/static/results/result_yolo.jpg',
                'result_custom': '/static/results/result_custom.jpg',
                'detections_yolo': detection_results_yolo if detection_results_yolo else [],
                'detections_custom': detection_results_custom,
                'success': True
            })

        except Exception as e:
            return jsonify({
                'error': f'Error al procesar la imagen: {str(e)}',
                'success': False
            })

@socketio.on('message')
def handle_message(message):
    timestamp = datetime.now().strftime('%H:%M')
    emit('message', {'msg': message, 'timestamp': timestamp}, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, host='127.0.0.1', port=5000)