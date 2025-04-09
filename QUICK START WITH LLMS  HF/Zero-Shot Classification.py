from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "I want to buy a new GPU for training large models."
labels = ["education", "hardware", "sports", "finance"]

result = classifier(text, candidate_labels=labels)
print(result)
# Output: Best label is likely "hardware"
