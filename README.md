# Emotion and Trigger Detection in Conversational Data Using BERT Models

This project focuses on detecting emotions and triggers within conversational data, leveraging BERT-based models for natural language understanding. 
The dataset used is MELD, containing dialogues from the sitcom FRIENDS, annotated with emotions and conversational triggers.

## Overview

We explored two main approaches:

1. A single BERT model with multiple output heads for emotion and trigger detection.
2. Separate BERT models for emotion and trigger detection, with one focusing on individual utterances and the other considering the entire dialogue context.

Our findings indicate that the second approach yields superior results, with emotion detection performing well without the need for contextual information from surrounding utterances.

## Libraries Used

```
numpy
pandas
matplotlib
transformers
torch
sklearn
```

## Data Preprocessing

We preprocess the MELD dataset by tokenizing textual data using a BERT tokenizer, creating a suitable format for our models. The data is split into training, validation, and testing sets with an 80-10-10 ratio.

## Models

### Emotion Detection Model

A BERT-based model (BERTClass) with a dropout layer and dense layers is employed to transform BERT output into emotion predictions. Class weights are computed to handle imbalances in emotion distribution.

### Trigger Detection Model

Two BERT models are utilized:
- One for identifying emotions in isolated utterances.
- Another for detecting triggers in the entire conversation, incorporating emotions identified by the first model.

### Training and Evaluation

Models are trained using the Adam optimizer with a linear learning rate scheduler. Performance is evaluated based on accuracy, precision, recall, F1-score, and validation loss.

## Results

Our experiments revealed:
- The second model architecture outperforms the first.
- Emotion detection does not significantly benefit from contextual information from surrounding utterances.
- Even with a smaller model like tiny BERT, stability in hyperparameter sensitivity is observed, unlike with larger models where variations in learning rates can introduce instability.

## Conclusion

This project demonstrates the effectiveness of BERT-based models in detecting emotions and triggers within conversational data. The separate model approach for emotion and trigger detection provides insights into the complexities of natural language understanding in dialogues.

## References

- HunEmBERT: A Fine-Tuned BERT Model for Classifying Sentiment and Emotion in Political Communication
- Fine-Tuning DistilBERT for Emotion Classification
- Knowledge-based BERT word embedding fine-tuning for emotion recognition
- BERT 101-State Of The Art NLP Model Explained
- Why AdamW matters

---

For more details on the implementation, please refer to the code and report provided in this repository.
