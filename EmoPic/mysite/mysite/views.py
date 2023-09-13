from django.shortcuts import render, HttpResponseRedirect, reverse
import requests
from django.conf import settings
from django.http import HttpResponse
import base64
import time
import uuid
from django.http import JsonResponse
import json
import os
import cv2
from PIL import Image
from django.http import HttpRequest
import requests
from django.http import HttpResponse
import base64
import time
import subprocess
import os
from django.http import FileResponse
from django.http import HttpResponseServerError
from .main import run_stargan


service_view_naver_OCR_api_url = # Place your API keys here
service_view_naver_OCR_secret_key = # Place your API keys here
screenshotlayer_capture_screenshot_api_key = # Place your API keys here
service_result_view_removebg_API_key = # Place your API keys here

def dashboard_view(request):
    return render(request, "dashboard.html")

def service_view(request):
    response_text = None

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        api_url = service_view_naver_OCR_api_url
        secret_key = service_view_naver_OCR_secret_key

        request_json = {
            'images': [
                {
                    'format': 'jpg',  # Change this format if needed
                    'name': 'demo'
                }
            ],
            'requestId': str(uuid.uuid4()),
            'version': 'V2',
            'timestamp': int(round(time.time() * 1000))
        }

        payload = {'message': json.dumps(request_json).encode('UTF-8')}
        files = [
            ('file', image)
        ]
        headers = {
            'X-OCR-SECRET': secret_key
        }

        response = requests.post(api_url, headers=headers, data=payload, files=files)
        response_data = response.json()

        # Extract 'inferText' information from the API response
        infer_text_list = [field['inferText'] for image in response_data['images'] for field in image['fields']]

        # Join the 'inferText' strings into a single string
        response_text = ' '.join(infer_text_list)

        request.session['response_text'] = response_text  # Save the response result in the session

    return render(request, "service.html", {'response_text': response_text})



def service_select_view(request: HttpRequest):
    predicted_sentence = request.GET.get('predicted_sentence', '첫페이지 기본값')
    return render(request, "service_image.html", {'predicted_sentence': predicted_sentence})


def upload_image(request):
    output_dir = settings.STATICFILES_DIRS[1]
    uploaded_image = request.FILES.get('image_input')
    if not uploaded_image:
        return HttpResponseServerError('No image uploaded.')

    # Read the content of the TemporaryUploadedFile
    image_data = uploaded_image.read()

    image_path = process_image(image_data)  # Pass image data, not the file object
    input_dir = settings.STATICFILES_DIRS[2]
    face_detection_and_resizing(input_dir, output_dir)
    return render(request, 'service_image.html')


def capture_screenshot(url):
    api_key = screenshotlayer_capture_screenshot_api_key
    viewport = '1440x2000'
    width = '1440'

    api_url = f'http://api.screenshotlayer.com/api/capture?access_key={api_key}&url={url}&viewport={viewport}&width={width}'
    response = requests.get(api_url)

    if response.status_code == 200:
        screenshot_data = response.content
        return screenshot_data
    else:
        return None

def process_image(image_data):
    image_filename = "input_image.png"
    image_path = os.path.join(settings.STATICFILES_DIRS[2], image_filename)

    with open(image_path, "wb") as out:
        out.write(image_data)

    img = Image.open(image_path)
    new_image_path = os.path.join(settings.STATICFILES_DIRS[2], "input_image.png")
    img.save(new_image_path, "PNG")

    return new_image_path

def face_detection_and_resizing(input_dir, output_dir):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    face_padding = 0.22
    crop_size = 128

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            try:
                img = cv2.imread(img_path)

                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for i, (x, y, w, h) in enumerate(faces):
                        padding_x = int(w * face_padding)
                        padding_y = int(h * face_padding)
                        expanded_x = max(0, x - padding_x)
                        expanded_y = max(0, y - padding_y)
                        expanded_w = min(img.shape[1], x + w + padding_x) - expanded_x
                        expanded_h = min(img.shape[0], y + h + padding_y) - expanded_y

                        face_roi = img[expanded_y:expanded_y + expanded_h, expanded_x:expanded_x + expanded_w]
                        eyes = eye_cascade.detectMultiScale(face_roi)

                        if len(eyes) >= 2:
                            resized_face = cv2.resize(face_roi, (crop_size, crop_size))
                            output_path = os.path.join(output_dir, f'{filename[:-4]}.png')
                            cv2.imwrite(output_path, resized_face)
                        else:
                            print(f"Insufficient eyes detected in face {i + 1} from {filename}. Skipping.")
                    if len(faces) == 0:
                        print(f"No faces detected in {filename}.")
                else:
                    print(f"Error reading {filename}.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def screenshot_view(request):
    if request.method == 'POST':
        url = request.POST.get('url')

        # Step 1: Capture Screenshot
        screenshot_data = capture_screenshot(url)

        if screenshot_data:
            # Step 2: Process Image
            image_path = process_image(screenshot_data)

            # Step 3: Face Detection and Resizing
            input_dir = settings.STATICFILES_DIRS[2]
            output_dir = settings.STATICFILES_DIRS[1]

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            face_detection_and_resizing(input_dir, output_dir)

            # Step 4: Encode and Render
            screenshot_url = 'data:image/png;base64,' + base64.b64encode(screenshot_data).decode('utf-8')
            resized_image = os.path.join(settings.STATICFILES_DIRS[1], "input_image.png")

            with open(resized_image, "rb") as img_file:
                img_data = img_file.read()

            encoded_img_data = base64.b64encode(img_data).decode('utf-8')
            resized_image_url = 'data:image/png;base64,' + encoded_img_data
            predicted_sentence = request.session.get('predicted_sentence', None)

            return render(request, 'service_image.html',
                          {'screenshot_url': screenshot_url, 'predicted_sentence': predicted_sentence, 'resized_image_url': resized_image_url})
        else:
            return HttpResponse('Failed to capture screenshot.')

    return render(request, 'screenshot.html')


def service_result_view(request):
    predicted_sentence = request.session.get('predicted_sentence', None)
    context = {'predicted_sentence': predicted_sentence}

    if request.method == 'POST':
        image_path = os.path.join(settings.BASE_DIR, 'stargan/results/result.jpg')

        with open(image_path, 'rb') as image_file:
            api_key = service_result_view_removebg_API_key

            response = requests.post(
                'https://api.remove.bg/v1.0/removebg',
                files={'image_file': image_file},
                data={'size': 'auto'},
                headers={'X-Api-Key': api_key},
            )

            if response.status_code == requests.codes.ok:
                image_filename = 'no-bg.png'
                image_path = os.path.join(settings.STATICFILES_DIRS[0], image_filename)
                with open(image_path, 'wb') as out:
                    out.write(response.content)
                preview_image = image_path
                context = {'preview_image': preview_image, 'predicted_sentence': predicted_sentence}

            else:
                error_message = "Error: {} - {}".format(response.status_code, response.text)


    return render(request, 'service_result.html', context)




from .kobert import predict, get_pytorch_kobert_model, torch, BERTClassifier


bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
device = torch.device("cpu")

model = BERTClassifier(bertmodel, dr_rate=0.5)
model.load_state_dict(torch.load('mysite/surprise.pt', map_location=torch.device('cpu')))
model.eval()


def predict_emotion(request):
    input_text = None
    predicted_sentence = None

    if request.method == 'POST':
        input_text = request.POST.get('text_input')
        predicted_sentence = predict(input_text)
        request.session['predicted_sentence'] = predicted_sentence  # Store in session

    context = {'input_text': input_text, 'predicted_sentence': predicted_sentence}
    return render(request, 'service.html', context)


def run_main(request):
    emotion = request.GET.get('emotion', '중립')
    run_stargan(emotion)
    predicted_sentence = request.session.get('predicted_sentence', None)
    return render(request, 'service_result.html', {'predicted_sentence': predicted_sentence})


def serve_image(request, filename):
    image_path = os.path.join(os.path.dirname(__file__), '../stargan/results', filename)
    return FileResponse(open(image_path, 'rb'), content_type='image/jpeg')
