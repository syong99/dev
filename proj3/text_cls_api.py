from fastapi import FastAPI, Form

# STEP 1. import modules
from transformers import pipeline

# STEP 2. create inference instance
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

app = FastAPI()


@app.post("/text/")
async def text(text: str = Form()):

    # STEP 3. prepare input data
    # text = "실적이 안좋다."


    # STEP 4. inference
    result = classifier(text)

    # STEP 5. visualize
    # print(result)
    
    return {"result": result}