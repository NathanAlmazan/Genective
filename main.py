import os
import logging
import keyboard
import numpy as np
import tensorflow as tf
import tensorflow_text as text

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)  # suppress tensorflow logs except for errors


##################### MODEL CONFIGURATIONS #####################
NUM_LAYERS = 4  # number of stacked decoder layers
D_MODEL = 128  # size of each word embeddings
DFF = 512  # size of feed feed forward network first layer
NUM_HEADS = 8  # number of attention heads
DROPOUT_RATE = 0.1  # for training purposes only
VOCAB_SIZE = 7931 # number of tokens or words the model knows
MAX_TOKENS = 128

################################################################


"""
    Tokenizer: Converts words into tokens represented 
    by a 64-bit integer identifier.
"""
bert_tokenizer_params = dict(lower_case=True)
tokenizer = text.BertTokenizer('./notebooks/vocab.txt', **bert_tokenizer_params)


"""
    Positional Encoding: Embeds the sequential information 
    of text into word embeddings.
"""
def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
    

"""
    Masked Self-Attention: Self-attention captures relationships 
    and dependencies between different words in a sequence, allowing 
    the model to weigh the importance of each word with respect to the 
    other words in the sequence.
"""
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        
        
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
 
"""
    Feed Forward Layer: A simple neural network with two layers.
    The first layer have 512 neurons while the second one have 256 neurons.
    There is a relu activation function in between.
"""   
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x
    
    
"""
    Decoder Layer: Decoder is the heart of Generative Pre-Trained 
    Transformers. The transformer decoder is capable of generating sequences 
    auto regressively, meaning it produces one next word at a time.
"""
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.causal_self_attention(x=x)
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                        dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x)

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x
    
    
"""
    Transformer model: The main model class which stacks four decoder layer 
    and a final linear layer that has softmax activation function. In simple words,
    the final layer computes the most probable next word.
"""
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=target_vocab_size,
                            dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, x):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.

        x = self.decoder(x)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        # Return the final output and the attention weights.
        return logits


######################## MAIN PROCESS #########################

# Initialize transformer model
transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        target_vocab_size=VOCAB_SIZE,
        dropout_rate=DROPOUT_RATE)
    
# Loads the weights
transformer.load_weights('./model_v2/checkpoints')  


def generate_text(sentence, maxlen=MAX_TOKENS):
    output_array = tokenizer.tokenize(sentence).merge_dims(-2, -1).to_tensor()
    query_size = output_array.shape[1]
    
    for i in range(maxlen):
        prediction = transformer(output_array, training=False)
        prediction = prediction[:, -1:, :]
        prediction = tf.argmax(prediction, axis=-1)
        output_array = tf.concat([output_array, prediction], axis=1)
        
        if prediction[0][0].numpy() == 3:
            break
        
    output = output_array[:, query_size:]
    output = tokenizer.detokenize(output).to_tensor()
    output = ' '.join([word.decode('utf-8') for word in output.numpy()[0]])
    return output


def __main__():
    print('\nWelcome to Genebot!!! Type a query and press "q" or "ctrl + C" to exit.\n')
    
    while True:
        query = input('You: ')
        print('Genebot: ...')
        answer = generate_text(query)
        print(answer)
        print('\n')
        
        if keyboard.is_pressed('q'):
            print("Thank you for using Genebot.")
            break
    

__main__()  # Run main program
