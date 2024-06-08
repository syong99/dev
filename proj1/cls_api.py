import PIL.Image
from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models\\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=99)
classifier = vision.ImageClassifier.create_from_options(options)


app = FastAPI()

import io
import PIL
import numpy as np

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    byte_file = await file.read()

    # STEP 3: Load the input image.
    # image = mp.Image.create_from_file(byte_file[0])

    # convert char array to binary array
    image_bin = io.BytesIO(byte_file) # jpg 파일을 읽은거

    # create PIL Image form binary array
    pil_img = PIL.Image.open(image_bin) #이미지 디코딩 압축된걸 압축 풀어준다 생각

    # Convert MP Image from PIL IMAGE
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    
    # STEP 4: Classify the input image.
    classification_result = classifier.classify(image)
    print(classification_result)

    # STEP 5: Process the classification result. In this case, visualize it.
    count = 10
    results = []
    for i in range(count):
        category = classification_result.classifications[0].categories[i]
        results.append({"category":category.category_name, "score": category.score})
    # result = f"{top_category.category_name} - ({top_category.score:.2f})"

    return {"result": results}