import logging
import sys
import requests
from io import BytesIO
from fastapi import FastAPI, Request
import csv
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          DetrImageProcessor, DetrForObjectDetection, CLIPProcessor, CLIPModel)
import torch
import json
from PIL import Image, ImageDraw
import base64


# Logging
def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            '%(name)s [%(asctime)s] [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    return logger


logger = get_logger('snowpark-container-service')

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f'cuda.is_available(): {torch.cuda.is_available()}')
logger.info(f'cuda.device_count(): {torch.cuda.device_count()}')
logger.info(f'device: {device}')

logger.info('Loading Data ...')
with open('/model_workspace/whitelist_model_class.csv', newline='') as f:
    reader = csv.reader(f)
    data = [item for sublist in reader for item in sublist]
logger.info('Finished Loading Data.')

google_large = "google/owlv2-large-patch14-ensemble"
logger.info(f'Loading Model {google_large}...')
model_google_large = AutoModelForZeroShotObjectDetection.from_pretrained(google_large).to(device)
processor_google_large = AutoProcessor.from_pretrained(google_large)
logger.info(f'Finished Loading Model {google_large}...')

google_base = "google/owlv2-base-patch16-ensemble"
logger.info(f'Loading Model {google_base}...')
model_google_base = AutoModelForZeroShotObjectDetection.from_pretrained(google_base).to(device)
processor_google_base = AutoProcessor.from_pretrained(google_base)
logger.info(f'Finished Loading Model {google_base}...')

openai_large = 'openai/clip-vit-large-patch14'
logger.info(f'Loading Model {openai_large}...')
model_openai_large = CLIPModel.from_pretrained(openai_large).to(device)
processor_openai_large = CLIPProcessor.from_pretrained(openai_large)
logger.info(f'Finished Loading Model {openai_large}...')

openai_base = 'openai/clip-vit-base-patch32'
logger.info(f'Loading Model {openai_base}...')
model_openai_base = CLIPModel.from_pretrained(openai_base).to(device)
processor_openai_base = CLIPProcessor.from_pretrained(openai_base)
logger.info(f'Finished Loading Model {openai_base}...')

facebook_large = 'facebook/detr-resnet-101'
logger.info(f'Loading Model {facebook_large}...')
model_facebook_large = DetrForObjectDetection.from_pretrained(facebook_large, revision="no_timm").to(device)
processor_facebook_large = DetrImageProcessor.from_pretrained(facebook_large, revision="no_timm")
logger.info(f'Finished Loading Model {facebook_large}...')

facebook_base = 'facebook/detr-resnet-50'
logger.info(f'Loading Model {facebook_base}...')
model_facebook_base = DetrForObjectDetection.from_pretrained(facebook_base, revision="no_timm").to(device)
processor_facebook_base = DetrImageProcessor.from_pretrained(facebook_base, revision="no_timm")
logger.info(f'Finished Loading Model {facebook_base}...')


def detect_google(
        im,
        data,
        model,
        processor
):
    inputs = processor(text=data, images=im, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes = torch.tensor([im.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

    scores = results['scores']
    labels = results['labels']

    # List to keep track of unique labels
    unique_labels = []
    unique_scores = []

    # Sort scores and labels while ensuring no duplicate labels
    for score, label in sorted(zip(scores, labels), reverse=True, key=lambda x: x[0]):
        if label.item() not in unique_labels:  # Ensure no duplicate labels
            unique_labels.append(label.item())
            unique_scores.append(score.item())
        if len(unique_labels) == 3:  # Stop after finding top 3 unique labels
            break

    # Convert label indices to actual data indices and retrieve the values
    top3_data = [{"label": data[label % len(data)], "score": score} for label, score in
                 zip(unique_labels, unique_scores)]
    json_data = json.dumps(top3_data)
    return json_data


@app.post("/detect-google-large", tags=["Endpoints"])
async def detect_google_large(request: Request):
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    for index, im_file in request_body:
        im_byte = requests.get(im_file).content
        im = Image.open(BytesIO(im_byte))
        im = im.convert('RGB')
        result = detect_google(im, data, model_google_large, processor_google_large)
        return_data.append([index, result])
    return {"data": return_data}


@app.post("/detect-google-base", tags=["Endpoints"])
async def detect_google_base(request: Request):
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    for index, im_file in request_body:
        im_byte = requests.get(im_file).content
        im = Image.open(BytesIO(im_byte))
        im = im.convert('RGB')
        result = detect_google(im, data, model_google_base, processor_google_base)
        return_data.append([index, result])
    return {"data": return_data}


def detect_openai(
        im,
        data,
        model,
        processor
):
    inputs = processor(text=data, images=im, return_tensors="pt", padding=True).to(device)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)

    # Pair text queries with their respective probabilities
    results = list(zip(data, probs[0].tolist()))

    # Sort results by probability in descending order and take the top 3
    top_3_results = sorted(results, key=lambda x: x[1], reverse=True)[:3]

    # Print top 3 results
    # Create a list of dictionaries
    top3_data = [{"label": text, "score": prob} for text, prob in top_3_results]

    # Convert the list of dictionaries to a JSON string
    json_data = json.dumps(top3_data)

    # Print the JSON string
    return json_data


@app.post("/detect-openai-large", tags=["Endpoints"])
async def detect_openai_large(request: Request):
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    for index, im_file in request_body:
        im_byte = requests.get(im_file).content
        im = Image.open(BytesIO(im_byte))
        im = im.convert('RGB')
        result = detect_openai(im, data, model_openai_large, processor_openai_large)
        return_data.append([index, result])
    return {"data": return_data}


@app.post("/detect-openai-base", tags=["Endpoints"])
async def detect_openai_base(request: Request):
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    for index, im_file in request_body:
        im_byte = requests.get(im_file).content
        im = Image.open(BytesIO(im_byte))
        im = im.convert('RGB')
        result = detect_openai(im, data, model_openai_base, processor_openai_base)
        return_data.append([index, result])
    return {"data": return_data}


def detect_facebook(
        im,
        model,
        processor
):
    inputs = processor(images=im, return_tensors="pt").to(device)
    outputs = model(**inputs)

    target_sizes = torch.tensor([im.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Store unique labels and their highest score
    unique_results = {}

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        # Only keep the highest score for each label
        if label_name not in unique_results or score.item() > unique_results[label_name]:
            unique_results[label_name] = score.item()

    # Sort by score and keep the top 3
    top_3_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)[:3]

    # Create the final list of dictionaries
    top3_data = [{"label": label, "score": score} for label, score in top_3_results]

    # Print the result
    json_data = json.dumps(top3_data)

    # Print the JSON string
    return json_data


@app.post("/detect-facebook-large", tags=["Endpoints"])
async def detect_facebook_large(request: Request):
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    for index, im_file in request_body:
        im_byte = requests.get(im_file).content
        im = Image.open(BytesIO(im_byte))
        im = im.convert('RGB')
        result = detect_facebook(im, model_facebook_large, processor_facebook_large)
        return_data.append([index, result])
    return {"data": return_data}


@app.post("/detect-facebook-base", tags=["Endpoints"])
async def detect_facebook_base(request: Request):
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    for index, im_file in request_body:
        im_byte = requests.get(im_file).content
        im = Image.open(BytesIO(im_byte))
        im = im.convert('RGB')
        result = detect_facebook(im, model_facebook_base, processor_facebook_base)
        return_data.append([index, result])
    return {"data": return_data}


# -------------------------------------------------------------------------------
def all_detect_google(
        im,
        data,
        model,
        processor
):
    inputs = processor(text=data, images=im, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes = torch.tensor([im.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

    scores = results['scores']
    labels = results['labels']
    boxes = results['boxes']

    # Lists to keep track of unique labels, scores, and boxes
    unique_labels = []
    unique_scores = []
    unique_boxes = []

    # Sort scores and labels while ensuring no duplicate labels
    for score, label, box in sorted(zip(scores, labels, boxes), reverse=True, key=lambda x: x[0].item()):
        if label.item() not in unique_labels:  # Ensure no duplicate labels
            unique_labels.append(label.item())
            unique_scores.append(score.item())
            unique_boxes.append(box.tolist())  # Convert box tensor to a list for easier handling
        if len(unique_labels) == 3:  # Stop after finding top 3 unique labels
            break

    # Draw the boxes on the image
    draw = ImageDraw.Draw(im)
    for box in unique_boxes:
        # Draw the rectangle (box)
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)

    # Save the image with boxes drawn in memory as base64
    buffered = BytesIO()
    im.save(buffered, format="PNG")
    boxed_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Convert label indices to actual data indices and retrieve the values
    top3_data = [{"label": data[label % len(data)], "score": score, "boxed_image": boxed_image_base64} for label, score
                 in
                 zip(unique_labels, unique_scores)]

    # Convert to JSON and print
    json_data = json.dumps(top3_data)
    return json_data


@app.post("/all-detect-google-large", tags=["Endpoints"])
async def all_detect_google_large(request: Request):
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    for index, im_file in request_body:
        im_byte = requests.get(im_file).content
        im = Image.open(BytesIO(im_byte))
        im = im.convert('RGB')
        result = all_detect_google(im, data, model_google_large, processor_google_large)
        return_data.append([index, result])
    return {"data": return_data}


@app.post("/all-detect-google-base", tags=["Endpoints"])
async def all_detect_google_base(request: Request):
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    for index, im_file in request_body:
        im_byte = requests.get(im_file).content
        im = Image.open(BytesIO(im_byte))
        im = im.convert('RGB')
        result = all_detect_google(im, data, model_google_base, processor_google_base)
        return_data.append([index, result])
    return {"data": return_data}


def all_detect_facebook(
        im,
        model,
        processor
):
    inputs = processor(images=im, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([im.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Store unique labels and their highest score
    unique_results = {}

    # Iterate over the scores, labels, and boxes
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]

        # Only keep the highest score for each label, including the box
        if label_name not in unique_results or score.item() > unique_results[label_name]["score"]:
            unique_results[label_name] = {
                "score": score.item(),
                "box": box.tolist()  # Convert the tensor to a list for JSON serialization
            }

    # Sort by score and keep the top 3
    top_3_results = sorted(unique_results.items(), key=lambda x: x[1]["score"], reverse=True)[:3]

    # Draw the boxes on the image
    draw = ImageDraw.Draw(im)
    for _, result in top_3_results:
        box = result["box"]
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)

    # Save the image with boxes drawn in memory as base64
    buffered = BytesIO()
    im.save(buffered, format="PNG")
    boxed_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Create the final list of dictionaries including label, score, and boxed image
    top3_data = [{"label": label, "score": data["score"], "boxed_image": boxed_image_base64} for label, data in
                 top_3_results]

    # Convert the final list of dictionaries to a JSON string
    json_data = json.dumps(top3_data)

    # Print the JSON string
    return json_data

@app.post("/all-detect-facebook-large", tags=["Endpoints"])
async def all_detect_facebook_large(request: Request):
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    for index, im_file in request_body:
        im_byte = requests.get(im_file).content
        im = Image.open(BytesIO(im_byte))
        im = im.convert('RGB')
        result = all_detect_facebook(im, model_facebook_large, processor_facebook_large)
        return_data.append([index, result])
    return {"data": return_data}


@app.post("/all-detect-facebook-base", tags=["Endpoints"])
async def all_detect_facebook_base(request: Request):
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    for index, im_file in request_body:
        im_byte = requests.get(im_file).content
        im = Image.open(BytesIO(im_byte))
        im = im.convert('RGB')
        result = all_detect_facebook(im, model_facebook_base, processor_facebook_base)
        return_data.append([index, result])
    return {"data": return_data}
