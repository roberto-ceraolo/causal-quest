import os
import numpy as np
import evaluate
import gc, torch, torch.distributed
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import set_seed
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer

set_seed(42)

gc.collect()
torch.cuda.empty_cache()

RUN_NAME = "flan-t5-xl-20-ep-24g-lora-new"
MODEL_NAME = "google/flan-t5-xl"

os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_PROJECT"] = "" 


dataset = load_dataset("json", data_files="causalquest13k_labeled_causal_prompt_iteration_6_gpt-4-turbo-2024-04-09.jsonl")["train"]
labels = dataset["is_causal"]
dataset = dataset.add_column("label", labels)
dataset = dataset.class_encode_column("label")


dataset = dataset.train_test_split(test_size=0.1, stratify_by_column="label", seed=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize_function(examples):
    return tokenizer(examples["query"], padding="max_length", max_length=512, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2
)

model.config.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, 
                            lora_config)


training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple) :
      logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    # output_dir="test_trainer", 
    evaluation_strategy="steps", 
    num_train_epochs=5, 
    logging_steps = 30, 
    report_to="wandb", 
    eval_steps=100,
    save_steps=100,
    run_name = RUN_NAME,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    save_total_limit=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()


