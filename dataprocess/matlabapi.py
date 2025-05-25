import matlab.engine
import numpy as np
from flask import Flask, abort, jsonify, request

app = Flask(__name__)

eng = matlab.engine.start_matlab()
eng.addpath('./matlab')

WHITELIST = ['127.0.0.1']

@app.before_request
def limit_remote_addr():
    if request.remote_addr not in WHITELIST:
        abort(403)

@app.route('/perlin_noise', methods=['POST'])
def perlin_noise():
    data = request.json
    x = data.get('x', 500)
    y = data.get('y', 500)
    iterations = data.get('iterations', [2,3,4,5,6,7,8,9,10])
    saturation = data.get('saturation', 0.0100)

    x = matlab.double(x)
    y = matlab.double(y)
    iterations = matlab.double(iterations)
    saturation = matlab.double(saturation)

    GeneralPerlinNoise = eng.Perlin_Noise(x,y,iterations,saturation)
    GeneralPerlinNoise = np.array(GeneralPerlinNoise)
    
    return jsonify(GeneralPerlinNoise.tolist())

@app.route('/perlin_noise_sparse', methods=['POST'])
def perlin_noise_sparse():
    data = request.json
    x = data.get('x', 500)
    y = data.get('y', 500)
    iterations = data.get('iterations', [1,2,3])
    saturation = data.get('saturation', 0.0100)

    x = matlab.double(x)
    y = matlab.double(y)
    iterations = matlab.double(iterations)
    saturation = matlab.double(saturation)

    GeneralPerlinNoise = eng.Perlin_Noise_Sparse(x,y,iterations,saturation)
    GeneralPerlinNoise = np.array(GeneralPerlinNoise)
    
    return jsonify(GeneralPerlinNoise.tolist())

@app.route('/imgaussfilt', methods=['POST'])
def imgaussfilt():
    data = request.json
    Im_list = data.get('Im')
    gaussFiltImage = data.get('gaussFiltImage', 0.6667)

    Im = np.array(Im_list)

    Im = matlab.double(Im)
    gaussFiltImage = matlab.double(gaussFiltImage)

    imgauss = eng.imgaussfilt(Im,gaussFiltImage)
    imgauss = np.array(imgauss)
    
    return jsonify(imgauss.tolist())

@app.route('/awgn', methods=['POST'])
def awgn():
    data = request.json
    Im_list = data.get('Im')
    SNR = data.get('SNR', 45)

    Im = np.array(Im_list)

    Im = matlab.double(Im)
    SNR = matlab.double(SNR)

    im_1 = eng.awgn(Im,SNR)
    im_1 = np.array(im_1)
    
    return jsonify(im_1.tolist())

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)