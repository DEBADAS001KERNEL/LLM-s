from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

theme='''A large language modelis a type of artificial intelligence algorithm that applies neural network techniques with lots of parameters to process and understand human languages or text using self-supervised learning techniques. Tasks like text generation, " \
machine translation, summary'''
q= "what is a large language model ? "
result = qa(question=q,context=theme)
print(result)

