## Fine-tuning LLM DistilBERT for :smile::heart_eyes: Emotion Classification :pensive::rage: using Parameter Efficient Fine-Tuning (PEFT)

The project's aim was to fine-tune the LLM DistilBERT to specialize on categorizing emotions in texts into five categories:

0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise

This was achieved by using Supervised Learning and a PEFT method, namely Low Rank Adaptation (LoRA). The advantage of this training approach is that it drastically reduces the storage requirements and computation costs as it adjusts only a small number of additional parameters while leaving the majority of the LLM parameters unchanged from their initial state. (1)
In particular, LoRA integrates a compact trainable submodule into the transformer architecture, while maintaining the pre-trained model weights, and integrating trainable rank decomposition matrices within every layer. (2)

## Prerequisites 
- [Python 3.11](https://www.python.org/downloads/release/python-3110/)
- Jupyter Notebook ```pip install notebook ```
- [Required dependencies](https://github.com/KatTiel/Fine-tuning_DistilBERT/blob/main/requirements.txt) ```pip install -r requirements.txt ```
- ['emotion' dataset from Hugging Face](https://huggingface.co/datasets/dair-ai/emotion) (3)
- GPU is recommended, e.g. using [Google Colab](https://colab.google) or [Kaggle Notebooks](https://www.kaggle.com/)

## Data Set & Preprocessing
The data set was split into a **training set** (80%, 16000 records), a **validation set** (10%, 2000 records) and a **test set** (10%, 2000 records).

Furthermore, the dataset was **tokenized**, so that words were represented as numbers and therefore could be fed into the computations.

## Pre-trained LLM DistilBERT 
DistilBERT is a smaller and lighter version of BERT with reduced computational costs. It underwent pretraining through a self-supervised approach on the same dataset, leveraging the BERT base model as a guidance. This method entails training solely on raw texts, without human annotations, thereby enabling the utilization of vast amounts of publicly accessible data. Inputs and labels are generated automatically from the texts via the BERT base model. (4)

## Performance Measurement
### Accuracy
Accuracy is a good general performance parameter when all classes are equally important.

<img width="161" alt="" src="https://github.com/KatTiel/stroke_binary_classification_CNN/assets/76701992/7417c4b4-09d8-4dba-bb11-8e9e9dbebc1e">

:heavy_exclamation_mark:This model reached an evaluation accuracy of **0.89**

## License
[MIT](https://choosealicense.com/licenses/mit/)

## References 
(1) Xu et al. Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment. arXiv:2312.12148 cs.CL [https://doi.org/10.48550/arXiv.2312.12148](https://doi.org/10.48550/arXiv.2312.12148)

(2) Hu et al., LoRA: Low-rank adaptation of large language models., in Proc. Int. Conf. Learn. Representations, 2022.

(3) DAIR.AI. emotion, Retrieved 3/2024 from [Hugging Face](https://huggingface.co/datasets/dair-ai/emotion)

(4) Sanh et al., DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108. cs.CL. [https://doi.org/10.48550/arXiv.1910.01108](https://doi.org/10.48550/arXiv.1910.01108)

