import os
import pandas as pd
import tensorflow as tf
import numpy as np
import gradio as gr
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from keras.metrics import Precision, Recall, CategoricalAccuracy
from gradio import components
from gradio import Interface
from keras.layers import TextVectorization

# Set the base path to the folder containing the data
base_path = r"C:\Users\User1\SentimentAnalysisData"  # Replace this with path to your folder

# Read the training data from a CSV file
df = pd.read_csv(os.path.join(base_path, 'train.csv'))

# Display the first few comments from the CSV file
df.head()

# Extract the comment text and the corresponding labels
X = df['comment_text']
y = df[df.columns[2:]].values

# Define the maximum number of words in the vocabulary
MAX_FEATURES = 200000

# Create a text vectorization layer
vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=1800,
    output_mode='int'
)

# Adapt the vectorizer to the comment text data
vectorizer.adapt(X.values)

# Vectorize the comment text
vectorized_text = vectorizer(X.values)

# Create a TensorFlow dataset from the vectorized text and labels
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))

# Cache the dataset for better performance
dataset = dataset.cache()

# Shuffle the dataset
dataset = dataset.shuffle(160000)

# Batch the dataset
dataset = dataset.batch(16)

# Prefetch the dataset for better performance
dataset = dataset.prefetch(8)

# Split the dataset into training, validation, and testing sets
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

# Create a sequential model
model = Sequential()

# Add an embedding layer to the model
model.add(Embedding(MAX_FEATURES+1, 32))

# Add a bidirectional LSTM layer to the model
model.add(Bidirectional(LSTM(32, activation='tanh')))

# Add fully connected layers as feature extractors
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

# Add a final sigmoid layer for classification
model.add(Dense(6, activation='sigmoid'))

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='BinaryCrossentropy', optimizer='Adam')

# Display the model summary
model.summary()

# Train the model
history = model.fit(train, epochs=20, validation_data=val) # Change number of epochs if you want. Note more epochs = longer training time

# Plot the training history
plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()

# Define an input comment
input_text = ['You freaking suck! I am going to hit you!']

# Vectorize the input comment
input_text = vectorizer(input_text)

# Make predictions on the input comment
res = model.predict(input_text)

# Convert the predictions to binary values (0 or 1) based on a threshold of 0.5
(res > 0.5).astype(int)

# Get a batch of test data
batch_X, batch_y = test.as_numpy_iterator().next()

# Make predictions on the batch of test data
(model.predict(batch_X) > 0.5).astype(int)

# Get the shape of the predictions
res.shape

# Create metrics objects for precision, recall, and accuracy
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

# Iterate over the test dataset and evaluate the model
for batch in test.as_numpy_iterator():
    # Unpack the batch
    X_true, y_true = batch
    
    # Make predictions on the batch
    yhat = model.predict(X_true)

    # Flatten the predictions and true labels
    y_true = y_true.flatten()
    yhat = yhat.flatten()

    # Update the metrics with the current batch
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

# Print the precision, recall, and accuracy
print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

# Save the model to a file
model.save('sentimentAnalysis.h5') # Name this whatever you want

# Load the saved model from a file
model = tf.keras.models.load_model('sentimentAnalysis.h5')

# Define an input string
input_str = vectorizer('I hate you!')

# Make predictions on the input string
res = model.predict(np.expand_dims(input_str,0))

# Print the predictions
res

# Define a function to score a comment
def score_comment(comment):
    # Vectorize the comment
    vectorized_comment = vectorizer([comment])
    
    # Make predictions on the vectorized comment
    results = model.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(df.columns[2:]):
        # Format the results as a text
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)

    return text

# Create a Gradio interface to interact with the scoring function
interface = components.Interface(
    fn=score_comment,
    inputs=components.Textbox(lines=2, placeholder='Comment to score'),
    outputs='text'
)

# Launch the interface
interface.launch(share=False)
