import os
import numpy as np
import evaluate
import gc, torch, torch.distributed

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import set_seed
from transformers import TrainingArguments, Trainer

set_seed(42)


gc.collect()
torch.cuda.empty_cache()

RUN_NAME = "phi-1.5-new-lora"
MODEL_NAME = "microsoft/phi-1_5"

os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_PROJECT"] = "" 



dataset = load_dataset("json", data_files="causalquest13k_labeled_causal_prompt_iteration_6_gpt-4-turbo-2024-04-09.jsonl")["train"]
labels = dataset["is_causal"]
dataset = dataset.add_column("label", labels)
dataset = dataset.class_encode_column("label")


dataset = dataset.train_test_split(test_size=0.1, stratify_by_column="label", seed=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

def tokenize_function(examples):
    return tokenizer(examples["query"], padding="longest", max_length=512, truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2
)

model.config.pad_token_id = tokenizer.eos_token_id


from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
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
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=1,
    gradient_accumulation_steps=1,
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


