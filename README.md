### Text-generation-with-Recurrent-Neural-Network
Horoscope predictions always come across as quite random to me - they almost always leave you wondering what exactly they mean, with phrases
like "Don't dwell on the negatives." or "Because the Sun has been moving through your opposite sign of Capricorn some people have been rather demanding in recent weeks. But others have come to your rescue time and again, and now it’s your turn to do something good for them.", do they actually make an impact to your life when you don't really know what they are talking about?

Anyways, in this project, I tried to follow Tensorflow's documentations and text generation examples and train a Recurrent Neural Network to 
generate Horoscope style advice/ predictions. I am following this guide from Tensorflow: https://www.tensorflow.org/text/tutorials/text_generation

The first step is to import the packages and data. We will be using mainly tensorflow. The data is a csv file containing 12946 horoscope predictions in the column "Fortune".
```import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab import drive
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import os
import time
drive.mount('/content/drive')
```

### Import data
```path = "/content/drive/MyDrive/Colab Notebooks/RNN with fortune/fortunes.csv"
data = pd.read_csv(path)
data = data.drop(columns=['Unnamed: 1']).dropna()
data = data.sample(frac=1).reset_index(drop=True)
print(len(data))
data.head(3)
```
### Preprocessing data to convert words into numbers
Then we create a tf dataset containing data entries.
```
raw_train_ds = tf.data.Dataset.from_tensor_slices(data["Fortune"])
```
We create dictionaries for all the words and encode them into integers, i.e. "When" -> 13, "You"-> 2, etc. We create train and test datasets containg tensors such as [[1,432,23,16,68]]
The target data will be the next words, e.g. If the sample data is "Your luck is good", then input is "Your luck is" and target is "luck is good". When integer encoded, they will both be tensors of integers of course.
```
text_dataset = tf.data.Dataset.from_tensor_slices(data["Fortune"])
max_features = 10000  # Maximum vocab size.
max_len = 150  # Sequence length to pad the outputs to.

# Create the layer.
vectorize_layer = tf.keras.layers.TextVectorization(
 max_tokens=max_features,
 output_mode='int',
 output_sequence_length=max_len,
 standardize=None)

# Now that the vocab layer has been created, call `adapt` on the text-only
# dataset to create the vocabulary. You don't have to batch, but for large
# datasets this means we're not keeping spare copies of the dataset.
vectorize_layer.adapt(text_dataset.batch(64))
before = vectorize_layer.get_vocabulary()
print(before[:5])
after = before.copy()
after.remove("[UNK]")
print(after[:5])
def vectorize_text(text):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text)
print(vectorize_text(next(iter(raw_train_ds))))
print(next(iter(raw_train_ds)))
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=after)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=after, invert=True)
ids_from_chars.get_vocabulary()[:5]
chars_sample = chars_from_ids(vectorize_text(next(iter(raw_train_ds))))
print(chars_sample)
#convert raw data (words) into integer coded data (numbers)
ds = raw_train_ds.map(vectorize_text)
#create input and target for each data sample
def split_input_target(sequence):
    input_text = sequence[0][:-1]
    target_text = sequence[0][1:]
    return input_text, target_text

dataset = ds.map(split_input_target)
```

Now that preprocessing is over, we split data into train, val, test data for training and testing.
```
def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset, len(data), shuffle=False)
```

### Train the model
```
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

dataset
# Length of the vocabulary in chars
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024
class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x
model = MyModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)
model.summary()
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
import os
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
model.compile(optimizer='adam', loss=loss)
EPOCHS = 5
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
```
### Predictions
Now that the model is trained, we can generate horoscope style sentences by giving the first few words to the model and asking it to generate the following words until it reahces a period.
We actually do get some pretty decent results. For example, when I gave it the word "Try", it gave me the following sentence: Try not to get impatient with people and do whatever it takes to get along with them today.
Although the phrase "Try not to get impatient with people" is in the data, it has never seen this exact sentence before in the data sample.
```
class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs. ["HK HSBC"] -> ["HK","HSBC"]
    x = tf.strings.split(
    inputs, sep=" ", maxsplit=-1, name=None)
    #["HK","HSBC"] -> [14,15]
    input_ids = ids_from_chars(x).to_tensor()
    #print(input_ids)

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
import time
start = time.time()
states = None
next_char = ["Try"]
result = [next_char]
#print(len(next_char[0].split()))
end = 0
for n in range(50):
  if end == 0:
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)
    try: 
      if next_char.numpy()[0].decode('utf-8')[-1] == ".":
        end = 1
    except:
      print(next_char.numpy()[0].decode('utf-8'))
      end = 1
  #print(next_char.numpy()[0].decode('utf-8'))
result = tf.strings.join(result,separator=' ')
end = time.time()
#print(result[0])
#print('\nRun time:', end - start)
result.numpy()[0].decode('utf-8')
```
Finally, we save the weights of the model so we can reuse it in the future.
```
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
# Save the weights
model.save_weights('/content/drive/MyDrive/Colab Notebooks/RNN with fortune/my_check_point')
#create new model as before and compile it as before
new_mod.compile(optimizer='adam', loss=loss)
# Restore the weights
new_mod.load_weights('/content/drive/MyDrive/Colab Notebooks/RNN with fortune/my_check_point')
```


