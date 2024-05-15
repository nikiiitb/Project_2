import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoModel , AutoImageProcessor , AutoTokenizer 
from datasets import load_dataset , Dataset
from PIL import Image
import torch
from transformers import BertTokenizer , BertModel
from torch import nn

torch.cuda.empty_cache()

# import os
# os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

def training_img_address(input):
    IMAGE_DIR = "./train2014_3d"
    input["image_id"] = f"{IMAGE_DIR}/{input['image_id']}"
    return input

SAMPLES = 10000

df_acc = pd.read_pickle("./vqa_v2_acc.pkl")
dataset = Dataset.from_pandas(df_acc)
dataset = dataset.remove_columns(['__index_level_0__'])
dataset = dataset.select(range(0,SAMPLES))
dataset = dataset.map(training_img_address)

X_img = np.array([np.array(Image.open(i).resize((640 , 480))) for i in dataset["image_id"]])
X_text = np.array(dataset["question"])

import itertools

labels = [item['ids'] for item in dataset['label']]
flattened_labels = list(itertools.chain(*labels))
unique_labels = list(set(flattened_labels))

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()} 

def replace_ids(inputs):
    '''Converting everything to one-hot-encoding'''
    h = [0 for i in id2label]
    for i in range(len(inputs["label"]["ids"])):
        t = inputs["label"]["ids"][i]
        w = inputs["label"]["weights"][i]
        if w > 0.5: w = 1
        else: w = 0.3
        # print(t , w)
        # print(label2id.get(t , 0))
        h[label2id.get(t , 0)] = w
    inputs["label"] = h
    return inputs


flat_dataset = dataset.map(replace_ids)
# flat_dataset = dataset.flatten()
NUM_CLASSES = np.array(flat_dataset["label"]).shape[1]
print("Number of Classes : " , NUM_CLASSES)

vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
bert_processor = BertTokenizer.from_pretrained('bert-base-uncased')

from transformers import TFAutoModelForSequenceClassification , TFAutoModel , TFAutoModelForImageClassification , TFBertModel
from tensorflow.keras.optimizers import Adam

# Load and compile our model
# tf_model = tf.keras.models.Sequential(
#     [
#         TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased"),
#         tf.keras.layers.Dense(1 , input_shape=(768,) , activation='sigmoid')
#     ]
# )

# tf_model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")

MAX_LEN = 40

y_test = np.array(flat_dataset["label"])
X_img_tf = vit_processor(X_img , return_tensors="np")["pixel_values"]
X_text_tf = bert_processor(X_text.tolist() , padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors="np")

vit_tf_model = TFAutoModel.from_pretrained("google/vit-base-patch16-224")
bert_tf_model = TFBertModel.from_pretrained('bert-base-uncased')

def build_model():
    vit_inputs = tf.keras.layers.Input(shape=(3, 224, 224))
    x_vit = vit_tf_model(vit_inputs).pooler_output
    
    bert_input1 = tf.keras.layers.Input(shape=(MAX_LEN,) , dtype=tf.int32)
    bert_input2 = tf.keras.layers.Input(shape=(MAX_LEN,) , dtype=tf.int32)
    x_bert = bert_tf_model(input_ids=bert_input1 , attention_mask=bert_input2).pooler_output
    
    # x = np.concatenate([x_vit , x_bert])
    x = tf.keras.layers.Concatenate(axis=1)([x_vit, x_bert])
    # x = x_bert
    
    x = tf.keras.layers.Dense(128 , input_shape=(768*2,) , activation='relu')(x)
    x = tf.keras.layers.Dense(64 , activation='relu')(x)
    x = tf.keras.layers.Dense(NUM_CLASSES , activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=[vit_inputs , bert_input1 , bert_input2] , outputs=x)
    
    return model


modelx = build_model()
modelx.compile(optimizer=Adam(3e-5) , loss="categorical_crossentropy", metrics=['accuracy'])
modelx.summary()

x_input_ids = X_text_tf["input_ids"]
x_attention_masks = X_text_tf["attention_mask"]

modelx.fit([X_img_tf , x_input_ids , x_attention_masks] , y_test , epochs=20 , batch_size=1)
