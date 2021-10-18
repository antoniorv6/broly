from PatchProcessing import PatchGenerator
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Input, Add, MultiHeadAttention, Dropout, LayerNormalization, Average
from PatchProcessing import *
from tensorflow.keras.models import Model

PATCH_SIZE = 64
IMAGE_SIZE = 224

def MLP(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def get_crossedAttentionLayer(x_input, crossed_ref, embedding_dim, 
                              n_heads, ff_units, dropout=0.1):
    
    x = LayerNormalization(epsilon=1e-9)(x_input)
    x = MultiHeadAttention(num_heads=n_heads,
                           key_dim=embedding_dim, dropout=dropout)(x_input, crossed_ref)
    x = Add()([x_input, x])
    x_ln = LayerNormalization(epsilon=1e-9)(x)
    x_ln = MLP(x_ln, ff_units, dropout)
    x_ln = Dense(embedding_dim)(x_ln)
    x_ln = Dropout(dropout)(x_ln)
    x = Add()([x_ln, x])

    return x

def get_BROLY_encoder_CrossAttention(input_shape, target_shape, embedding_dim, num_layers, n_heads, ff_layers, last_ff_layers, out_features):
    input_in = Input(shape=input_shape, name="source_img")
    input_tar = Input(shape=target_shape, name="source_img")

    n_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2

    x_inputPatches = PatchGenerator(PATCH_SIZE)(input_in)
    x_input_enc = PatchEncoder(n_patches, embedding_dim)(x_inputPatches)

    x_targetPatches = PatchGenerator(PATCH_SIZE)(input_tar)
    x_tar_enc = PatchEncoder(n_patches, embedding_dim)(x_targetPatches)

    last_input = x_input_enc
    tar_enc = x_tar_enc
    in_enc = last_input
    
    for _ in range(num_layers): 
        in_enc = get_crossedAttentionLayer(in_enc, tar_enc, embedding_dim, n_heads, ff_layers)
        tar_enc = get_crossedAttentionLayer(tar_enc, last_input, embedding_dim, n_heads, ff_layers)
        last_input = in_enc
    
    ## We have crossed all, now we have to merge the last embeddings
    x_avg = Average()([in_enc, tar_enc])
    x = MLP(x_avg, last_ff_layers)
    x = Add()([x, x_avg])

    output = Dense(out_features, activation='sigmoid')(x)

    model = Model([input_in, input_tar], output)
    model.compile(optimizer = "adam", loss="binary_crossentrpy") 


    




