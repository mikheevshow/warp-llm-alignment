"""
This file contains all required code for model training
"""

import numpy as np

from typing import Tuple, Optional, Dict, Any

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedModel
)

from datasets import (
    load_dataset, 
    load_metric
)


DATASET_NAME = "imdb"
DISTILBERT_UNCASED = "distilbert/distilbert-base-cased"
warp_pretrained_reward_model_name = "warp-reward-model"


def load_trained_warp_reward_model(owner: Optional[str] = "mikheevshow") -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Loads pretrained warp reward model. By default it loads a model from repo `mikheevshow`
    """
    tokenzer = AutoTokenizer.from_pretrained(DISTILBERT_UNCASED)
    if owner is not None:
        full_model_name = f"{owner}/{warp_pretrained_reward_model_name}"
    else:
        full_model_name = warp_pretrained_reward_model_name

    model = AutoModelForSequenceClassification.from_pretrained(full_model_name)
  
    return tokenzer, model


def compute_metrics(eval_pred) -> Dict[str, Any]:
    """
    Computes accuracy and f1 metrics. Uses with evaluation of sentiment classification
    """
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]

    return {"accuracy": accuracy, "f1": f1}


def create_train_reward_model(publish: bool = False, proportion_of_train: float = 0.12, proportion_of_eval: float = 0.12) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Train reward model 
    """
    assert proportion_of_train > 0.0 and proportion_of_train <= 1.0
    assert proportion_of_eval > 0.0 and proportion_of_eval <= 1.0

    # Prepare train and evaluation datasets
    print(f"Loading {DATASET_NAME} dataset")
    imdb = load_dataset(DATASET_NAME)

    def sample(ds, ds_type):
        num_rows = ds.num_rows[ds_type]
        num_rows_for_sample = int(num_rows * (proportion_of_train if ds_type == 'train' else proportion_of_eval))
        return ds[ds_type].shuffle(seed=42).select([i for i in list(range(num_rows_for_sample))])

    reduced_train_dataset = sample(imdb, 'train')
    reduced_test_dataset = sample(imdb, 'test')

    print("Start load tokenizer")
    tokenzer = AutoTokenizer.from_pretrained(DISTILBERT_UNCASED)

    def preprocess_function(examples):
        return tokenzer(examples["text"], truncation=True)
    
    tokenized_train = reduced_train_dataset.map(preprocess_function, batched=True)
    tokenized_test = reduced_test_dataset.map(preprocess_function, batched=True)

    print("Start load model")
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_UNCASED)
    collator = DataCollatorWithPadding(tokenizer=tokenzer)

    # Create trainer and train reward model
    training_args = TrainingArguments(
        output_dir=f"{warp_pretrained_reward_model_name}/",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch"
    )
 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenzer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Start train...")
    trainer.train()
    print("Start evaluation...")
    trainer.evaluate()

    print("WARP Reward model trainging completed")

    if publish:
        model.push_to_hub(warp_pretrained_reward_model_name)

    return tokenzer, model