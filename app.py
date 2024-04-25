import os
import io
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import SS_model

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'PNG', 'JPG'}
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360  # 新たに追加
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)
threshold = 127

# SS_modelをロード
model = SS_model.load_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def object_detection(img):
    set = img
    set2 = cv2.cvtColor(set, cv2.COLOR_BGR2RGB)
    wai = np.asarray(set2, dtype=np.float64)[np.newaxis, :, :, :]
    predictions = model.predict(wai)
    for i in range(720):
        for j in range(1280):
            if predictions[0, i, j, 0] >= 0.5:
                set[i, j] = (0, 0, 0)
    return set

def calculate_displayed_pixels(detected_img_path):
    # 処理された画像の読み込み
    detected_img = cv2.imread(detected_img_path)

    # 処理された画像の高さと幅を取得
    height, width, _ = detected_img.shape

    # 処理された画像のピクセル数を計算
    total_pixels = height * width

    # 表示されている部分のピクセル数を計算
    displayed_pixels = cv2.countNonZero(cv2.cvtColor(detected_img, cv2.COLOR_BGR2GRAY))

    # 比率を計算
    ratio = displayed_pixels / total_pixels

    return ratio

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']

        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
        else:
            return ''' <p>許可されていない拡張子です</p> '''

        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        raw_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_'+filename)
        cv2.imwrite(raw_img_url, img)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'gray_'+filename)
        cv2.imwrite(gray_img_url, gray_img)
        
        # オブジェクト検出を適用
        detected_img = object_detection(img)
        detected_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_'+filename)
        cv2.imwrite(detected_img_url, detected_img)

        # アップロードされた画像のパスを渡して画面に表示
        ratio = calculate_displayed_pixels(detected_img_url)
        return render_template('index.html', raw_img_url=raw_img_url, detected_img_url=detected_img_url, ratio=ratio)

    else:
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.debug = True
    app.run()
