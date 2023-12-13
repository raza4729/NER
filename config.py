from transformers import TrainingArguments

# Please define the model here
# model = "distilbert-base-uncased"

# Please safely remore the following path and add path where you want to store the data from hugginface and models
path = "/mount/studenten/arbeitsdaten-studenten1/razaai/cache"

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