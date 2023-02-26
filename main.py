import numpy as np
import tensorflow as tf

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import TFAutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AdamWeightDecay
from transformers import pipeline
from transformers.keras_callbacks import KerasMetricCallback
import evaluate

tokenizer = AutoTokenizer.from_pretrained("t5-small")
sacrebleu = evaluate.load("sacrebleu")

source_lang = "en"
target_lang = "es"
prefix = "translate English to Spanish: "


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [prefix + example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result


if __name__ == '__main__':
    print("Hello world")

    books = load_dataset("opus_books", "en-es")
    books = books["train"].train_test_split(test_size=0.99)

    print(books["train"][0])

    tokenized_books = books.map(preprocess_function, batched=True)

    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

    # Get the pretrained model (Tensorflow)
    model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")

    tf_train_set = model.prepare_tf_dataset(
        tokenized_books["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_books["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        tokenized_books["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    model.compile(optimizer=optimizer)

    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_set)
    callbacks = [metric_callback]

    # Actually begin model training
    model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2, callbacks=callbacks)

    # Inference
    text = "translate English to Spanish: I would like to kiss my little duck."

    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    print(translator(text))

    # for item in tf_test_set:
    #     print(item)

    print("All done!")
