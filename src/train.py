import torch
import numpy as np
from datasets import load_metric
from sklearn.metrics import f1_score
from transformers import TrainingArguments
from transformers import ViTForImageClassification
from transformers import EarlyStoppingCallback
from transformers import Trainer
from data_preprocessing import device, feature_extractor, prepared_train, prepared_val #, prepared_test

#дата-коллатор
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

# f1 metric
metric = load_metric("f1")
def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average='weighted'
    )


#задаем аргументы для файнтюна
training_args = TrainingArguments(
  output_dir="/kaggle/working/",
  per_device_train_batch_size=32,
  evaluation_strategy="epoch",
  logging_strategy = "epoch",
  save_strategy =  "epoch",
  num_train_epochs=30,
#   save_steps=100,
#   eval_steps=100,
#   logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  load_best_model_at_end=True,
)

labels = {0:'front', 1:'left', 2:'other', 3:'right'}
model_id = 'google/vit-base-patch16-224-in21k'

#загружаем изначальную модель
model = ViTForImageClassification.from_pretrained(
    model_id,  # classification head
    num_labels=len(labels)
)
model.to(device)

#обратная связь для остановки после переобучения
early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

#трейнер для файнтюна
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_train,
    eval_dataset=prepared_val,
    callbacks = [early_stopping],
    tokenizer=feature_extractor
)

train_results = trainer.train()
# save tokenizer with the model
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
# save the trainer state
trainer.save_state()