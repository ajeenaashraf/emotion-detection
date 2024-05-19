from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = load_model('model.h5')
        label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
        categorized_results = {label: [] for label in label_map}
        print(request.files)

        # Handling manual file upload
        if 'manual_files' in request.files:
            files = request.files.getlist('manual_files')
            for file in files:
                if file.filename == '':
                    continue
                filename = secure_filename(file.filename)
                filepath = os.path.join('static', filename)
                file.save(filepath)
                process_image(filepath, model, categorized_results, label_map)

        # Handling folder selection
        if 'select_files' in request.files:
            files = request.files.getlist('select_files')
            if not files or files[0].filename == '':
                return "No files selected"
            
            for file in files:
                if file.filename == '':
                    continue
                if os.path.isdir(file.filename):
                    folder_path = secure_filename(file.filename)
                    for filename in os.listdir(folder_path):
                        filepath = os.path.join(folder_path, filename)
                        process_image(filepath, model, categorized_results, label_map)
                else:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join('static', filename)
                    file.save(filepath)
                    process_image(filepath, model, categorized_results, label_map)

        # Return the prediction results
        return render_template('predict.html', categorized_results=categorized_results)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def process_image(filepath, model, categorized_results, label_map):
    try:
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 3)
        
        for x, y, w, h in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = image[y:y + h, x:x + w]

            marked_image_path = os.path.join('static', 'marked_' + os.path.basename(filepath))
            cv2.imwrite(marked_image_path, image)

            img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))
            img = img / 255
            img = img.reshape(1, 48, 48, 1)

            pred = model.predict(img)
            pred = np.argmax(pred)
            final_pred = label_map[pred]

            categorized_results[final_pred].append({
                'filename': os.path.basename(filepath),
                'marked_image_path': marked_image_path,
            })
    except Exception as e:
        print(f"An error occurred while processing image {filepath}: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
