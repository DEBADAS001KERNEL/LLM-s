from transformers import pipeline

sumarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """Large Language Models (LLMs) represent a breakthrough in artificial intelligence, employing neural network techniques with extensive parameters for advanced language processing.

This article explores the evolution, architecture, applications, and challenges of LLMs, focusing on their impact in the field of Natural Language Processing (NLP).

What are Large Language Models(LLMs)?
A large language model is a type of artificial intelligence algorithm that applies neural network techniques with lots of parameters to process and understand human languages or text using self-supervised learning techniques. Tasks like text generation, machine translation, summary writing, image generation from texts, machine coding,
chat-bots, or Conversational AI are applications of the Large Language Model."""

result = sumarizer(text, max_length=40, min_length=10, do_sample=False) # we can change max_lenth, min_lenth etc.
print(result)

