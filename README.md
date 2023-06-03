# Sentiment Analysis Project

This project is a sentiment analysis model implemented using TensorFlow and Keras. The model is trained to predict the sentiment of text data and classify it into different categories.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)

## Overview

The sentiment analysis model in this project is designed to analyze text data and classify it into different sentiment categories. It utilizes a deep learning approach with an LSTM-based architecture to capture the sequential nature of text data.

The project includes functionality to preprocess the text data, build and train the sentiment analysis model, and provide an interactive interface to predict the sentiment of user-provided comments.

## Installation

To run this project, you need to have the following dependencies installed:

- Python (version 3.9 or higher)
- TensorFlow (version 2.0 or higher)
- Keras (version 2.3 or higher)
- Pandas
- NumPy
- Gradio
- Matplotlib

You can install the required dependencies by running the following command:

```shell
pip install -r requirements.txt

## Usage
To use the sentiment analysis model, follow these steps:

1. Prepare your labeled dataset: The project assumes that you have a labeled dataset in CSV format. Update the train.csv file with your dataset, where each row represents a comment and its corresponding sentiment labels.

2. Train the model: Run the model function to build and train the sentiment analysis model using the labeled dataset. The script will preprocess the text data, create the model architecture, compile it, and train it on the provided dataset.

3. Save the trained model: The trained model will be saved as sentimentAnalysis.h5 in the models/ directory. This file will be used for prediction and evaluation.

4. Evaluate the model: The score_comment function provides functionality to score individual comments using the trained model. You can use this script to evaluate the model's performance on test data or new comments.

5. Run the Gradio interface: Use the Gradio link that appears when the code is ran to test the model with a GUI

## Dataset
The sentiment analysis model requires a labeled dataset for training. The dataset should be in CSV format, with each row representing a comment and its corresponding sentiment labels. The dataset used in this project is from the Toxic Comment Classification Challenge on Kaggle:
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/rules

## Model-Architecture
The sentiment analysis model in this project is built using the following architecture:

Text preprocessing: The text data is tokenized and vectorized using a TextVectorization layer provided by TensorFlow.

Embedding Layer: The vectorized text data is passed through an Embedding layer to convert it into dense word embeddings.

Bidirectional LSTM: A Bidirectional LSTM layer is used to capture the sequential information and context in the text data.

Fully Connected Layers: The output from the LSTM layer is fed into fully connected layers to extract features and perform sentiment classification.

Final Layer: The final layer is a dense layer with sigmoid activation to produce the probability scores for each sentiment category.

The model is trained using the Binary Cross-Entropy loss and the Adam optimizer.

## Evaluation
The model's performance can be evaluated using precision, recall, and accuracy metrics. Feel free to train the model on more or less epochs than provided in the code
