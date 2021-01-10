from django.shortcuts import HttpResponse, render, redirect
from django.http import JsonResponse
import uuid
from django.core.files.base import ContentFile
from buah.libraries.feature_extraction_factory import FeatureExtractionFactory
import cv2
import pickle
import numpy as np
import json
import base64
import os
from os.path import join
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from buah.models import Dataset, Model
from django.conf import settings
from utils.helper import get_second_value, get_url_citra


def index(request):
    models = Model.objects.all()
    context = {
        'models': models
    }
    return render(request, 'index.html', context)


def index_lama(request):
    print(request.user)
    return render(request, 'index_lama.html')


def pengujian(request):
    return render(request, 'pengujian.html')

def process_pengujian(request):
    model_id = int(request.POST.get('model_id'))
    image64 = request.POST.get('image')
    formatt, imgstr = image64.split(';base64,')
    ext = formatt.split('/')[-1]
    filename = str(uuid.uuid4())
    fileImg = ContentFile(base64.b64decode(imgstr), name=filename + "." + ext)
    image = cv2.imdecode(np.fromstring(fileImg.read(), np.uint8), 1)
    factory = FeatureExtractionFactory()
    morfologi = factory.make_morfologi(image)
    image_gray = morfologi.gray
    image_binary = morfologi.cleaned.astype(int)*255

    image_clean = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
    image_gray = base64.b64encode(cv2.imencode('.jpg', image_gray)[1]).decode()
    image_binary = base64.b64encode(cv2.imencode('.jpg', image_binary)[1]).decode()

    glcm = factory.make_glcm(image)


    prd      =  morfologi.get_prd()
    plw      =  morfologi.get_plw()
    rect     =  morfologi.get_rect()
    nf       =  morfologi.get_narrow_factor()
    ar       =  morfologi.get_aspect_ratio()
    ff       =  morfologi.get_form_factor()

    mean_h      = glcm.get_mean_h()
    mean_s  = glcm.get_mean_s()
    mean_v      = glcm.get_mean_v()

    dataset = {}
    dataset['form_factor'] = ff
    dataset['aspect_ratio'] = ar
    dataset['rect'] = rect
    dataset['narrow_factor'] = nf
    dataset['prd'] = prd
    dataset['plw'] = plw
    dataset['mean_h'] = mean_h
    dataset['mean_s'] = mean_s
    dataset['mean_v'] = mean_v

    test_data = np.fromiter(dataset.values(), dtype=float).reshape(1, -1)

    model = Model.objects.get(id=model_id)
    model_path = model.filename.path

    with open(model_path, 'rb') as handle:
    	model_file = pickle.load(handle)

    scaler = model_file['scaler']
    model_anfis = model_file['model']
    test_data_scaled = scaler.transform(test_data)
    from buah.libraries.anfis_matlab import AnfisMatlab
    import matlab
    import matlab.engine
    engine = matlab.engine.find_matlab()[0]
    engine = matlab.engine.connect_matlab(engine)
    matlablibdir = join(settings.BASE_DIR, 'matlab')
    engine.cd(matlablibdir)
    am = AnfisMatlab(engine)
    predicted = am.mulai_pengujian(test_data_scaled, model_anfis)
    predicted = (int(predicted))
    variables = {
        "prd": prd,
        "plw": plw,
        "rect": rect,
        "nf": nf,
        "ar": ar,
        "ff": ff,
        "mean_h": mean_h,
        "mean_s": mean_s,
        "mean_v": mean_v,
    }
    kelas_hasil = predicted
    nilai_kelas = list(zip(*Dataset.KELAS_CHOICES))[0]
    if (predicted in nilai_kelas):
    	kelas_hasil = get_second_value(Dataset.KELAS_CHOICES, predicted)

    citra_prediksi = get_url_citra(predicted)

    response = {
        'success': 1,
        'predicted': predicted,
        'fitur': json.dumps(dataset),
        'fitur_scaled': list(test_data_scaled.reshape(-1)),
        'variables': variables,
        'image_clean': image_clean,
        'image_gray': image_gray,
        'image_binary': image_binary,
        'kelas_hasil': kelas_hasil,
        'citra_prediksi': citra_prediksi

    }

    return JsonResponse(response, safe=False)


def login(request):
    return render(request, 'login.html')


def process_login(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    user = authenticate(username=username, password=password)
    success = 0
    if user is not None:
        auth_login(request, user)
        success = 1
    context = {
        'success': success
    }
    return JsonResponse(context, safe=False)


def logout(request):
	auth_logout(request)
	return redirect('index')
