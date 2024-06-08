# STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2
app = FaceAnalysis(providers=['CPUExecutionProvider']) # CPUExecutionProvider CPU 가지고 돌린다 라는걸 명시
app.prepare(ctx_id=0, det_size=(640, 640))

# STEP 3
#img = ins_get_image('t1')
img = cv2.imread('images.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('images.jpg', cv2.IMREAD_COLOR)

# STEP 4
faces1 = app.get(img)
faces2 = app.get(img2)

# STEP 5
# then print all-to-all face similarity
# feats = []
# feats.append(faces[0].normed_embedding) # normed_embedding
# feats.append(faces[0].normed_embedding)

feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)

# feats = np.array(feats, dtype=np.float32)
sims = np.dot(feat1, feat2.T) # dot 512개의 1차원 배열을 행열 연산을 통해 비교
print(sims)

# rimg = app.draw_on(img, faces)
# cv2.imwrite("./dog_output.jpg", rimg)