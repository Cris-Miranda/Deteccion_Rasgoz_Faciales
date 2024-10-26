from flask import Flask, render_template, request, send_file
import os
import cv2
import dlib
from io import BytesIO

# Configurar flask y Google Drive
app = Flask(__name__)
UPLOAD_FOLDER = '/content/gdrive/MyDrive/Emotion+AI+Dataset/Emotion AI Dataset/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el detector de rostros y el predictor de puntos faciales de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/content/gdrive/MyDrive/Emotion+AI+Dataset/Emotion AI Dataset/shape_predictor_68_face_landmarks.dat')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Guardar la imagen en la carpeta especificada en Google Drive
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Procesar la imagen para detectar puntos faciales
            img = cv2.imread(filepath)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_img)

            if len(faces) > 0:
                face = faces[0]
                landmarks = predictor(gray_img, face)

                # Puntos específicos definidos por ti:
                left_eye_points = [36, 37, 38]  # ojo izquierdo
                right_eye_points = [42, 43, 44]  # ojo derecho
                left_brow_points = [17, 21]  # ceja izquierda
                right_brow_points = [22, 26]  # ceja derecha
                mouth_points = [48, 51, 54, 57]  # boca
                nose_point = [30]  # nariz

                def draw_x(img, x, y, size=8, color=(0, 0, 255), thickness=2):
                    """Dibuja una 'X' en el punto (x, y) con mayor tamaño."""
                    cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness)
                    cv2.line(img, (x + size, y - size), (x - size, y + size), color, thickness)

                # Dibujar las 'X' en los puntos faciales específicos
                for i in left_eye_points + right_eye_points + left_brow_points + right_brow_points + mouth_points + nose_point:
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    draw_x(img, x, y)  # Dibuja una "X" en los puntos

            # Redimensionar la imagen a 600x600 píxeles
            resized_img = cv2.resize(img, (600, 600))

            # Guardar la imagen redimensionada en un buffer
            _, buffer = cv2.imencode('.jpg', resized_img)
            img_io = BytesIO(buffer)

            # Mostrar la imagen con las "X" en los puntos faciales redimensionada a 600x600 px
            return send_file(img_io, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
