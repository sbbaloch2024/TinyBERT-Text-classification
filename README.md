# TinyBERT Fine-Tuning on IMDb Dataset  

This repository contains a script for fine-tuning [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) on the IMDb movie reviews dataset using PyTorch and the 🤗 Transformers library. The goal is to perform sentiment classification on IMDb reviews.  

## 📌 Features  
- Fine-tunes [TinyBERT (4L-312D)](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) for binary sentiment classification.  
- Implements efficient tokenization with `AutoTokenizer`.  
- Utilizes `Trainer` API from the 🤗 Transformers library for streamlined training.  
- Evaluates model performance using accuracy as a metric.  
- Saves the fine-tuned model and tokenizer for future use.  

## 🛠️ Installation  

Ensure you have Python installed and then install the required dependencies:  

```bash
pip install torch transformers datasets evaluate
