# STEP 1. import modules
from transformers import pipeline

# STEP 2. create inference instance
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model") # task 네임 와 model(hugging face 에 검색하면 나오고 설명도 나온다.)

# STEP 3. prepare input data
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# STEP 4. inference
result = classifier(text)

# STEP 5. visualize
print(result)