# Sentiment Analysis Project

This project is a sentiment analysis model implemented using TensorFlow and Keras. The model is trained to predict the sentiment of text data and classify it into different categories. Files include a Python file, a Jupyter Notebook file, and a requirements.txt file.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)

## Overview

The sentiment analysis model in this project is designed to analyze text data and classify it into different sentiment categories. It utilizes a deep learning approach with an LSTM-based architecture to capture the sequential nature of text data.

The project includes functionality to preprocess the text data, build and train the sentiment analysis model, and provide an interactive interface to predict the sentiment of user-provided comments.

## Installation

To run this project, you need to have the following dependencies installed:

- Python (version 3.9 or higher)

- TensorFlow: TensorFlow is an open-source machine learning framework. It provides a platform for building and training machine learning models, including deep learning models. In this project, TensorFlow is used for creating and training the sentiment analysis model.

- Keras: Keras is a high-level neural networks API written in Python and runs on top of TensorFlow. It provides an easy-to-use interface for building and training deep learning models. In this project, Keras is used for constructing the sentiment analysis model architecture and handling the training process

- Pandas: Pandas is a powerful data manipulation and analysis library. It provides data structures and functions to efficiently handle and manipulate structured data, such as CSV files. In this project, Pandas is used for reading the labeled dataset from a CSV file and performing data preprocessing.

- NumPy: NumPy is a fundamental library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. In this project, NumPy is used for various numerical computations and array operations.

- Gradio: Gradio is a Python library for quickly creating customizable UI components for machine learning models. It allows you to build interactive interfaces to make predictions or visualize results. In this project, Gradio is used to create an interactive interface for the sentiment analysis model, where users can input comments and get sentiment predictions.

- Matplotlib: Matplotlib is a popular plotting library in Python. It provides a wide range of functions and tools for creating visualizations and plots. In this project, Matplotlib is used for visualizing the training and validation loss history of the sentiment analysis model.

You can install the required dependencies by running the following command:

pip install -r requirements.txt


## Usage
To use the sentiment analysis model, follow these steps:

1. Prepare your labeled dataset: The project assumes that you have a labeled dataset in CSV format. Update the train.csv file with your dataset, where each row represents a comment and its corresponding sentiment labels.

2. Train the model: Run the model function to build and train the sentiment analysis model using the labeled dataset. The script will preprocess the text data, create the model architecture, compile it, and train it on the provided dataset.

3. Save the trained model: The trained model will be saved as sentimentAnalysis.h5 in the models/ directory. This file will be used for prediction and evaluation.

4. Evaluate the model: The score_comment function provides functionality to score individual comments using the trained model. You can use this script to evaluate the model's performance on test data or new comments.

5. Run the Gradio interface: The code to launch the Gradio interface is included in the code, so use the Gradio link that appears when the code is ran to test the model with a GUI. For more help with Gradio see this link: https://gradio.app/quickstart/
6

## Example
Example 1: (https://drive.google.com/file/d/1pxXS6mdqEzVBW91aiZl86rzlLpni62_k/view?usp=sharing)

Example 2: (https://drive.google.com/file/d/1RJ81-D39NOnkjDCI106ENKmae5Iu5uvr/view?usp=sharing)


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
