# vqa_test.py
# import tensorflow as tf
from transformers import ViTFeatureExtractor, ViTForImageClassification, BertTokenizer, TFBertModel
from transformers import ViltForQuestionAnswering , ViltProcessor , TrainingArguments , Trainer , DefaultDataCollator
import numpy as np
import pandas as pd
from datasets import load_dataset , Dataset
import itertools
from PIL import Image
from os import listdir
import torch

# #Loading the dataset
# print("Reading the dataset")
# df = pd.read_pickle("./vqa_v2.pkl")

# def weight(x):
#     return 1 if x == "yes" else 0.5

# def accumulate_labels(l):
#     h = {}
#     s = 0
#     for i in l:
#         r = h.get(i["answer"] , 0)
#         r += (inc:=weight(i["answer_confidence"]))
#         s += inc
#         h[i["answer"]] = r
#     for i in h:
#         h[i] =  round((h[i]/s) , 2)
#     return {"ids" : list(h.keys()) , "weights" : list(h.values())}

# def convert_gray2rgb(image):
#     width, height = image.shape
#     out = np.empty((width, height, 3), dtype=np.uint8)
#     out[:, :, 0] = image
#     out[:, :, 1] = image
#     out[:, :, 2] = image
#     return out

# IMAGE_DIR = "./train2014_3d"

df_acc = pd.read_pickle("./vqa_v2_acc.pkl")
dataset = Dataset.from_pandas(df_acc)
dataset = dataset.remove_columns(["__index_level_0__"])

labels = [item['ids'] for item in dataset['label']]
flattened_labels = list(itertools.chain(*labels))
unique_labels = list(set(flattened_labels))

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()} 

# def replace_ids(inputs):
#     inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
#     inputs["image_id"] = f"{IMAGE_DIR}/{inputs['image_id']}"
#     return inputs


# flat_dataset = dataset.map(replace_ids)
# flat_dataset = flat_dataset.flatten()
# flat_dataset = flat_dataset.remove_columns(['question_type', 'question_id', 'answer_type'])

# print("Created Flattened Dataset")

# from transformers import ViltProcessor

# model_checkpoint = "dandelin/vilt-b32-mlm"
# # try:
# #     processor = ViltProcessor.from_pretrained(model_checkpoint , device_map = 'cuda')
# #     print("Transferred model to GPU")
# # except:
# #     processor = ViltProcessor.from_pretrained(model_checkpoint)
# #     print("Model running on CPU")

# print("device_map = auto")
# processor = ViltProcessor.from_pretrained(model_checkpoint , device_map = 'auto')

# import torch

# def preprocess_data(examples):
#     image_paths = examples['image_id']
#     images = [Image.open(image_path) for image_path in image_paths]
#     texts = examples['question']    

#     max_length = 2048
#     encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt" , max_length=max_length)
#     for k, v in encoding.items():
#           encoding[k] = v.squeeze()
    
#     targets = []

#     for labels, scores in zip(examples['label.ids'], examples['label.weights']):
#         target = torch.zeros(len(id2label))

#         for label, score in zip(labels, scores):
#             target[label] = score
      
#         targets.append(target)

#     encoding["labels"] = targets
    
#     return encoding

# processed_dataset = flat_dataset.map(
#     preprocess_data, 
#     batched=True, 
#     batch_size=128,
#     writer_batch_size=128,
#     cache_file_name="./preprocess_data_cache",
#     # keep_in_memory=False
#     # remove_columns=['question','question_type',  'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights']
#     )

# print("Creating Processed Data")

# processed_dataset.save_to_disk("processed_dataset.hf")

# print("Saved Processed Dataset")

training_data1 = Dataset.load_from_disk("./processed_dataset_14_clean.hf")
# training_data1 = training_data1.select(range(0,10))
print(training_data1)

model_checkpoint = "dandelin/vilt-b32-mlm"
processor = ViltProcessor.from_pretrained(model_checkpoint)
model = ViltForQuestionAnswering.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)

print("Transferring to CUDA")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE : " , device.type)
model.to(device)

print("Init Data Collator")
data_collator = DefaultDataCollator()

# repo_id = "nikrath/vilt_finetuned_200"
repo_id = "./vilt_finetuned_14_test"

print("Init Training Arguments")
training_args = TrainingArguments(
    output_dir=repo_id,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    save_steps=200,
    logging_steps=50,
    learning_rate=5e-5,
    save_total_limit=2,
    # fp16=True if device.type == 'cuda' else False,
    remove_unused_columns=False,
    push_to_hub=False,
    # use_cpu = True,
)

print("Init Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=training_data1,
    tokenizer=processor,
)

print("Beginning Training")
trainer.train()
trainer.save_model("vilt_finetuned_14_test_model")