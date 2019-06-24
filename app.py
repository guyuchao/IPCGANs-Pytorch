from flask import Flask
from demo import Demo
from flask import Flask,request
from PIL import Image
from gevent.pywsgi import WSGIServer
import base64



app = Flask(__name__)
D=Demo("checkpoint/IPCGANS/2019-01-14_08-34-45/saved_parameters/gepoch_6_iter_4000.pth")

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/upload_file',methods=['POST'])
def upload_file():
    img = request.files['img']
    img.save("tmp.jpg")
    img=Image.open("tmp.jpg").resize((400,400),Image.BILINEAR)
    for i in range(5):
        D.demo(img,i).resize((400,400),Image.BILINEAR).save("tmp%d.jpg"%i)
    with open("tmp.jpg", "rb") as f:
        base64_data = base64.b64encode(f.read())
    return base64_data

@app.route('/get_image/<string:age>',methods=['GET'])
def get_image(age):
    with open("tmp%s.jpg"%age, "rb") as f:
        base64_data = base64.b64encode(f.read())
    return base64_data

if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
