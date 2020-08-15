import tensorflow as tf

'''
For each sequence, duplicate and shift it to form the input 
and target text by using the map method to apply a simple 
function to each batch:
'''
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

'''
Three layers are used for this simple model:
- tf.keras.layers.Embedding: The input layer. A trainable lookup table that will map the numbers of each 
    character to a vector with embedding_dim dimensions;
- tf.keras.layers.GRU: A type of RNN with size units=rnn_units (You can also use a LSTM layer here.)
- tf.keras.layers.Dense: The output layer, with vocab_size outputs.
'''
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model