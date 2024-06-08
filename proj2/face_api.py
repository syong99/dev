#STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image > 인사이트페이스에서 제공해주는 이미지

#STEP 2 추론기 할당
face = FaceAnalysis(providers=['CPUExecutionProvider'])
face.prepare(ctx_id=0, det_size=(640, 640))

########################################
from typing import Union
from fastapi import FastAPI, File, UploadFile
app = FastAPI()

from PIL import Image
import numpy as np
import io
@app.post("/files/")
async def create_file(file: bytes = File(),
                      file2: bytes = File()):

    #STEP 3
    # img = cv2.imread("twice.jpg", cv2.IMREAD_COLOR)
    nparr = np.fromstring(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    nparr2 = np.fromstring(file2, np.uint8)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
    
    # img = ins_get_image('t1')

    #STEP 4 추론결과
    result = face.get(img)
    result2 = face.get(img2)
    
    #예외처리
    if len(result) == 0 or len(result2) == 0:
        return {"result":"fail"}
    
    face1 = result[0]
    face2 = result2[0]
    
    #유사도 측정
    emb1 = face1.normed_embedding
    emb2 = face2.normed_embedding
    
    #임베딩 값을 비교하기위해서 np.dot메소드를 써서 sim 값에 대입
    sim = np.dot(emb1, emb2)

    #STEP 5 후처리로 비즈니스로직 처리
    return{"result": float(sim)}