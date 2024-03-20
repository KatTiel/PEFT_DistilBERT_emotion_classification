## Fine-tuning LLM DistilBERT for Emotion Classification using Parameter Efficient Fine-Tuning (PEFT)

The project's aim was to fine-tune the LLM DistilBERT to specialize on classifying emotions in texts into five categories:

0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise

This was achieved by using Supervised Learning and a PEFT method, namely Low Rank Adaptation (LoRA). The advantage of this training approach is that it is drastically reduces the storage requirements and computation costs as it adjusts only a small number of additional parameters while leaving the majority of the LLM parameters unchanged from their initial state.(1) 

## Prerequisites 
- [Python 3.11](https://www.python.org/downloads/release/python-3110/)
- Jupyter Notebook ```pip install notebook ```
- [Required dependencies](.............................) ```pip install -r requirements.txt ```
- ['emotion' dataset from Hugging Face](https://huggingface.co/datasets/dair-ai/emotion) (2)
- GPU is recommended, e.g. using [Google Colab](https://colab.google) or [Kaggle Notebooks](https://www.kaggle.com/)

## Data Set & Preprocessing
Hugging Face's 'emotion' dataset was utilized for the fine-tuning.

The data set was split into a **training set** (80%, 1600 records), a **validation set** (10%, 2000 records) and a **test set** (10%, 2000 records).

## Pre-trained LLM DistilBERT 

## Performance Measurements
### Accuracy
Accuracy is a good general performance parameter when all classes are equally important.

<img width="161" alt="" src="https://github.com/KatTiel/stroke_binary_classification_CNN/assets/76701992/7417c4b4-09d8-4dba-bb11-8e9e9dbebc1e">

### Loss 

## License
[MIT](https://choosealicense.com/licenses/mit/)

## References 
(1) Xu et al. Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment. arXiv:2312.12148 cs.CL [https://doi.org/10.48550/arXiv.2312.12148](https://doi.org/10.48550/arXiv.2312.12148)

(2) DAIR.AI. emotion, Retrieved 3/2024 from [Hugging Face](https://huggingface.co/datasets/dair-ai/emotion)