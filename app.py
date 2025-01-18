from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import os
from pathlib import PosixPath, WindowsPath
import pathlib

pathlib.PosixPath = pathlib.WindowsPath
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

# Configurar directorios fijos
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ORIGINAL_IMAGE = os.path.join(UPLOAD_FOLDER, 'original.jpg')
RESULT_IMAGE = os.path.join(UPLOAD_FOLDER, 'result.jpg')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar modelo YOLO
model = torch.hub.load('ultralytics/yolov5', 'custom', path='modelos/best.pt')


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


def process_image(image_path):
    # Cargar imagen original
    original_img = cv2.imread(image_path)

    # Realizar detección
    results = model(image_path)

    # Obtener detecciones
    detections = results.xyxy[0].cpu().numpy()  # xyxy format: x1, y1, x2, y2, confidence, class

    # Dibujar solo las cajas delimitadoras en la imagen original
    detected_diseases = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        # Convertir a enteros
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Obtener etiqueta y confianza
        label = model.names[int(cls)]
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
    cv2.imwrite(RESULT_IMAGE, original_img)

    return RESULT_IMAGE, detected_diseases


@app.route('/')
def home():
    return render_template('index.html')


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
            if os.path.exists(ORIGINAL_IMAGE):
                os.remove(ORIGINAL_IMAGE)
            if os.path.exists(RESULT_IMAGE):
                os.remove(RESULT_IMAGE)

            # Guardar nueva imagen original
            file.save(ORIGINAL_IMAGE)

            # Procesar imagen y obtener detecciones
            result_path, detected_diseases = process_image(ORIGINAL_IMAGE)

            # Preparar resultados de detección
            detection_results = []
            if detected_diseases:
                for disease, confidence in detected_diseases:
                    detection_results.append({
                        'disease': disease,
                        'confidence': f"{confidence * 100:.1f}%"
                    })
                    recommendation = get_treatment_recommendation(disease.lower(), confidence)
                    timestamp = datetime.now().strftime('%H:%M')
                    socketio.emit('message', {
                        'msg': f"Detección: {disease} (Confianza: {confidence:.2f})\n\n{recommendation}",
                        'timestamp': timestamp
                    })

            return jsonify({
                'original': '/static/uploads/original.jpg',
                'result': '/static/uploads/result.jpg',
                'detections': detection_results if detection_results else [],
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