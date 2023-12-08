from transformers import TrainingArguments

# label mappings
label2id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-ANIM": 7, "I-ANIM": 8,
    "B-BIO": 9, "I-BIO": 10, "B-CEL": 11, "I-CEL": 12, "B-DIS": 13, "I-DIS": 14, "B-EVE": 15, "I-EVE": 16,
    "B-FOOD": 17, "I-FOOD": 18, "B-INST": 19, "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23,
    "I-MYTH": 24, "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, "I-VEHI": 30,
}

# in order to convert leave-out labels to zero following list will be used as reference
blocked_labels = {"B-BIO": 9, "I-BIO": 10, "B-CEL": 11, "I-CEL": 12, "B-EVE": 15, "I-EVE": 16,
    "B-FOOD": 17, "I-FOOD": 18, "B-INST": 19, "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23,
    "I-MYTH": 24, "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, "I-VEHI": 30,
}

id2label = {v: k for k, v in label2id.items()}

# Please define the model here
model = "distilbert-base-uncased"

# Please safely remore the following path and add path where you want to store the data from hugginface and models
path = "/mount/studenten/arbeitsdaten-studenten1/razaai/cache"

# define hyperparameters
training_args = TrainingArguments(
    output_dir="named-entity-recognition-distilbert-B",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)