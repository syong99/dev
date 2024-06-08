# STEP 1. import modules
from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# STEP 2
# tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
# model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")

# STEP 2. create inference instance
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC") # task 네임 와 model(hugging face 에 검색하면 나오고 설명도 나온다.)

# STEP 3. prepare input data
text = "실적이 안좋다."

# STEP 4
# inputs = tokenizer(text, return_tensors="pt")
# with torch.no_grad():
#     logits = model(**inputs).logits

# STEP 4. inference
result = classifier(text)


# 전처리 4-1 . preprocessing(data 사람이 입력한 데이터 -> tensor(blob) 모델이 이해할 수 있는 데이터)
# 추론   4-2 . inference(tensor(blog) -> logit 풀어낸 데이터)
# 후처리 4-3 . postprocessing(logit -> data)

# STEP 5. visualize
print(result)