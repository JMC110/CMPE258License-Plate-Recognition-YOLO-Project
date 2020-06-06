import os

from flask import Flask, render_template, request, send_file

from dl_prediction.predictor import Predictor

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

predictor = Predictor()
output_car_file = 'static/output_car.jpg'
output_license_file = 'static/output_license.jpg'
output_license_original_file = 'static/output_license_original.jpg'
output_video_file = 'static/output_video.mp4'


@app.route('/')
@app.route('/yolo')
def yolo():
    return render_template("yolo.html")


@app.route('/yolo/video')
def yolo_video():
    return render_template("yolo_video.html")


@app.route('/yolo/upload', methods=['POST'])
def yolo_upload():
    isImage = request.args.get('type') == 'image'
    isVideo = request.args.get('type') == 'video'
    file = request.files['file']
    file.save(file.filename)
    if (isImage):
        license_txt = predictor.predict(file.filename, output_car_file, output_license_original_file, output_license_file)
    elif (isVideo):
        predictor.predict(file.filename, output_car_file, output_license_original_file, output_license_file, output_video_file, is_image=False)
    else:
        raise Exception("Unsupported file type")

    os.remove(file.filename)
    if isVideo:
        return render_template("yolo_video.html")
    else:
        return render_template("yolo.html", license_text=license_txt)


@app.route('/cnn')
def cnn():
    return render_template("cnn.html")


@app.route('/cnn/video')
def cnn_video():
    return render_template("cnn_video.html")


@app.route('/cnn/upload', methods=['POST'])
def cnn_upload():
    isImage = request.args.get('type') == 'image'
    isVideo = request.args.get('type') == 'video'
    file = request.files['file']
    file.save(file.filename)
    if (isImage):
        license_txt = predictor.predict(file.filename, output_car_file, output_license_original_file, output_license_file, is_cnn=True)
    elif (isVideo):
        predictor.predict(file.filename, output_car_file, output_license_original_file, output_license_file, output_video_file, is_cnn=True, is_image=False)
    else:
        raise Exception("Unsupported file type")

    os.remove(file.filename)
    if isVideo:
        return render_template("cnn_video.html")
    else:
        return render_template("cnn.html", license_text=license_txt)


if __name__ == '__main__':
    app.run()
