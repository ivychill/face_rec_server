#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0.0.01
@author: qingyao.wang
@license: kuang-chi Licence 
@contact: qingyao.wang@kuang-chi.com
@site: 
@software: PyCharm
@file: Server.py
@time: 2018/11/19 15:54
"""
from cStringIO import StringIO
import base64,cv2,zmq,msgpack
import msgpack_numpy as m
import numpy as np
from PIL import Image
from flask import Flask,  request, jsonify
app = Flask(__name__)

def image_to_base64(img):
    output_buffer = StringIO()
    img.save(output_buffer, format='JPEG')
    binary_data = output_buffer.getvalue()
    base64_data = base64.b64encode(binary_data)
    return base64_data

def getIdentifyPerson(id,img):
    #发送给检测服务器
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        try:
            message = [id,img]
            serialized = msgpack.packb(message, default=m.encode)
            socket.connect ('tcp://10.2.40.194:12111')
            socket.setsockopt(zmq.RCVTIMEO,2000)
            socket.send (serialized)
            recvmessage = socket.recv()
            messagelist = msgpack.unpackb(recvmessage, object_hook=m.decode)
            #print messagelist
            socket.close()
            context.term()
            return messagelist
        except Exception as e:
             print e
             socket.close()
             context.term()
    except:
        pass

    return None

@app.route('/recognition/', methods=['POST'])
def add_task():
    if request.json and 'id' in request.json and 'image' in request.json:
        #abort(400)
        try:
            img_data=base64.b64decode(request.json['image'])
            nparr = np.fromstring(img_data,np.uint8)
            img=cv2.imdecode(nparr,cv2.COLOR_BGR2RGB)
            id = request.json['id']

            # 识别
            try:
                infolist = getIdentifyPerson(id,img)
                #print infolist
                if infolist!=None \
                and len(infolist[1])!=0 \
                and len(infolist[2]) != 0 \
                and len(infolist[4]) != 0:
                    Name = infolist[1][0]
                    IDNumber = infolist[2][0]
                    IDPhoto = infolist[4][0]
                    #IDPhotoBase64 = image_to_base64(Image.fromarray(cv2.cvtColor(IDPhoto,cv2.COLOR_BGR2RGB)))
                    IDPhotoBase64 = image_to_base64(Image.fromarray(IDPhoto))
                    return jsonify({'result': 'success','Name':Name,"ID":IDNumber,'IDPhoto':IDPhotoBase64})
                else:
                    return jsonify({'result': 'failure2'})
            except:
                return jsonify({'result': 'failure3'})
        except:
            return jsonify({'result': 'decode error'})
    else:
        return jsonify({'result': 'failure'})



from multiprocessing import Process
class WebServer(Process):
    def __init__(self,addr,port):
        Process.__init__(self)
        self.addr = addr
        self.port = port

    def run(self):
        print('websrver process start!')
        app.run(host=self.addr, port=self.port,debug=False)


if __name__ == "__main__":
    app.run(host="192.168.10.76", port=8383, debug=True)
