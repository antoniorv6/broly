import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Dense, Input, Add, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, Conv2D, BatchNormalization, Dropout
from PatchProcessing import *
from TransformerMeta import *

patch_size = 64

def MLP(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


### ENCODER RELATED CODE ###

def GetBROLYFirstLayer(input_seq, target_seq,
                      embedding_dim, n_heads, ff_dim,
                      num_patches,
                      dropout_rate=0.1):
  
  ## Bloque MHA input
  x_inputPatches = PatchGenerator(patch_size)(input_seq)
  x_input_encoded = PatchEncoder(num_patches, embedding_dim)(x_inputPatches)
  x_in = LayerNormalization(epsilon=1e-6)(x_input_encoded)
  mha_xin = MultiHeadAttention(num_heads=n_heads, key_dim=embedding_dim, 
                               dropout=dropout_rate)(x_in, x_in) 
  x_in = Add()([mha_xin, x_input_encoded])
  x_in_ln = LayerNormalization(epsilon=1e-6)(x_in)
  x_in_ln = MLP(x_in_ln, ff_dim, dropout_rate)
  x_in = Add()([x_in_ln, x_in])
  ####

  ## Bloque MHA target
  x_targetPatches = PatchGenerator(patch_size)(target_seq)
  x_target_encoded = PatchEncoder(num_patches, embedding_dim)(x_targetPatches)
  x_tar = LayerNormalization(epsilon=1e-6)(x_target_encoded)
  mha_xtar = MultiHeadAttention(num_heads=n_heads, key_dim=embedding_dim, 
                                dropout=dropout_rate)(x_tar, x_tar)
  x_tar = Add()([mha_xtar, x_target_encoded])
  x_tar_ln = LayerNormalization(epsilon=1e-6)(x_tar)
  x_tar_ln = MLP(x_tar_ln, ff_dim, dropout_rate)
  x_tar = Add()([x_tar_ln, x_tar])
  ####

  # We do not want to skip this block
  x = MultiHeadAttention(num_heads=n_heads, key_dim=embedding_dim, 
                         dropout=dropout_rate, name='crossMHA')(x_in, x_tar)
  
  x_ln = LayerNormalization(epsilon=1e-6)(x)
  x_ln = MLP(x, ff_dim, dropout_rate)
  x = Add()([x_ln, x])
  x = LayerNormalization(epsilon=1e-6)(x)

  return x


def GetBROLYEncoderLayer(input_seq,
                        embedding_dim, n_heads, ff_dim,
                        dropout_rate=0.1):
  x = LayerNormalization(epsilon=1e-6)(input_seq)
  x_mha = MultiHeadAttention(num_heads=n_heads, 
                             key_dim=embedding_dim, dropout=dropout_rate)(x,x)
  x = Add()([x_mha, x])
  x_ln = LayerNormalization(epsilon=1e-6)(x)
  x_ln = MLP(x_ln, ff_dim, dropout_rate)
  x = Add()([x_ln, x])
  x = LayerNormalization(epsilon=1e-6)(x)

  return x

######

### DECODER RELATED CODE ###
def GetDecoderFirstLayer(context_vector, decoder_input, la_mask, 
               embedding_dim, n_heads, ff_dim, seq_len, dropout_rate=0.1):
  
  #projection = Dense(embedding_dim)(decoder_input)
  pos_embedding = tf.range(start=0, limit=embedding_dim, delta=1, dtype=tf.float32)
  embedding_decoder = Embedding(input_dim=seq_len, output_dim=embedding_dim)(decoder_input)
  embedding_decoder = embedding_decoder + pos_embedding

  ### First MHA layer
  x_mha = MultiHeadAttention(num_heads=n_heads, key_dim=embedding_dim, dropout=dropout_rate)(embedding_decoder, embedding_decoder, attention_mask=la_mask)
  x = Add()([x_mha, embedding_decoder])
  x_ln = LayerNormalization(epsilon=1e-6)(x)
  x_ln = MLP(x_ln, ff_dim, dropout_rate)
  x = Add()([x_ln, x])
  x = LayerNormalization(epsilon=1e-6)(x)
  ###

  ### Crossed Attention between
  x_mha_cross = MultiHeadAttention(num_heads=n_heads, key_dim=embedding_dim, dropout=dropout_rate)(x, context_vector)
  x = Add()([x_mha_cross, x])
  x_ln_cross = LayerNormalization(epsilon=1e-6)(x)
  x_ln_cross = MLP(x_ln_cross, ff_dim, dropout_rate)
  x = Add()([x_ln_cross, x])
  x = LayerNormalization(epsilon=1e-6)(x)

  
  return x


def GetDecoderLayer(context_vector, decoder_in, la_mask, 
               embedding_dim, n_heads, ff_dim, seq_len, dropout_rate=0.1):

  ### First MHA layer
  x_mha = MultiHeadAttention(num_heads=n_heads, key_dim=embedding_dim, dropout=dropout_rate)(decoder_in, decoder_in, attention_mask=la_mask)
  x = Add()([x_mha, decoder_in])
  x_ln = LayerNormalization(epsilon=1e-6)(x)
  x_ln = MLP(x_ln, ff_dim, dropout_rate)
  x = Add()([x_ln, x])
  x = LayerNormalization(epsilon=1e-6)(x)
  ###

  ### Crossed Attention between
  x_mha_cross = MultiHeadAttention(num_heads=n_heads, key_dim=embedding_dim, dropout=dropout_rate)(x, context_vector)
  x = Add()([x_mha_cross, x])
  x_ln_cross = LayerNormalization(epsilon=1e-6)(x)
  x_ln_cross = MLP(x_ln_cross, ff_dim, dropout_rate)
  x = Add()([x_ln_cross, x])
  x = LayerNormalization(epsilon=1e-6)(x)

  return x


def GetBROLYFirstStage(input_shape, target_shape, embedding_dim, ff_depth, attention_heads, combination_type, num_layers):
  input_in = Input(shape=input_shape, name="source_img")
  input_tar = Input(shape=target_shape, name="target_img")

  ##TODO - Set num patches by size with parameters
  n_patches = (224//64)**2 

  ## Encoder first layer
  x = GetBROLYFirstLayer(input_in, 
                         input_tar, 
                         embedding_dim, 
                         attention_heads, 
                         ff_depth, n_patches)
  
  hidden_layers = []
  last_hidden_layer = x
  for _ in range(num_layers):
    last_hidden_layer = GetBROLYEncoderLayer(last_hidden_layer, embedding_dim, attention_heads, ff_depth)
    hidden_layers.append(last_hidden_layer)
  
  print("[SUCCESS] - BROLY Encoder created")
  
  ### EMBEDDING CONCATENATION PHASE ###

  ## Embedding = Last Hidden Layer
  if combination_type == 0:
    x = Flatten()(last_hidden_layer)

  ## Embedding = Sum of all layers
  if combination_type == 1:
    x = Flatten()(Add()(hidden_layers))
  
  ## Embedding = Second-to-last hidden layer
  if combination_type == 2:
    x = Flatten()(hidden_layers[-2])
  
  ## Embedding = Sum of the last four hidden layers
  if combination_type == 3:
    x = Flatten()(Add()(hidden_layers[-4:]))
  
  if combination_type == 4:
    x = Flatten()(Concatenate()(hidden_layers[-4:]))

  ########

  # PREDICTION PHASE
  #flip_x = Dense(1, activation="sigmoid")(x)
  #flip_y = Dense(1, activation="sigmoid")(x)
  #zoom = Dense(1, activation="sigmoid")(x)
  #blur = Dense(1, activation="sigmoid")(x)
  #inversion = Dense(1, activation="sigmoid")(x)
  output = Dense(5, activation="sigmoid")(x)

  ## Now we have to construct or output for the BROLY encoder
  model = Model([input_in, input_tar], output)
  transformer_optimizer = Get_Custom_Adam_Optimizer(embedding_dim)
  model.compile(optimizer= transformer_optimizer, loss= 'categorical_crossentropy')
  model.summary()
  return model


###### FULL TRANSFORMER MODEL

def GetBROLYTransformer(input_shape, target_shape, seq_len, embedding_dim, ff_depth,
                       attention_heads, num_layers_encoder, num_layers_decoder, num_instructions):
  input_in = Input(shape=input_shape, name="Source image")
  input_tar = Input(shape=target_shape, name="Target image")
  input_decoder = Input(shape=(None,), name="In decoder")
  feed_forward_mask = Input(shape=(None,), name="FF")

  n_patches = (224 // 64) ** 2

  ## First BORT Layer
  x = GetBROLYFirstLayer(input_in, input_tar, embedding_dim, attention_heads, ff_depth, n_patches) 
  print(x.shape)
  for _ in range(num_layers_encoder):
    x = GetBROLYEncoderLayer(x, embedding_dim, attention_heads, ff_depth)

  print("BORT CREATED!")  
  context_vector = tf.identity(x)
  print(context_vector.shape)

  x = GetDecoderFirstLayer(context_vector, input_decoder, feed_forward_mask, embedding_dim, attention_heads, ff_depth, seq_len)

  for _ in range(num_layers_decoder):
    x = GetDecoderLayer(context_vector, x, feed_forward_mask, embedding_dim, attention_heads, ff_depth, seq_len)
  
  output = Dense(num_instructions, activation='softmax')(x)

  model = Model([input_in, input_tar, input_decoder, feed_forward_mask], output)

  transformer_optimizer = Get_Custom_Adam_Optimizer(embedding_dim)

  model.compile(optimizer= transformer_optimizer, loss= Transformer_Loss_AIAYN, metrics=['accuracy'])
  model.summary()
  return model


def GetBROLYTransformerCNN(input_shape, target_shape, seq_len, embedding_dim, ff_depth,
                       attention_heads, num_layers_encoder, num_layers_decoder, num_instructions):
  input_in = Input(shape=input_shape, name="Source image")
  input_tar = Input(shape=target_shape, name="Target image")
  input_decoder = Input(shape=(None,), name="In decoder")
  feed_forward_mask = Input(shape=(None,), name="FF")

  #CNN_BLOCK_INPUT
  cnn_in = Conv2D(128, (5,5),padding="same")(input_in)
  cnn_in = BatchNormalization()(cnn_in)
  cnn_in = ReLU()(cnn_in)
  cnn_in = Dropout(0.2)(cnn_in)
  cnn_in = Conv2D(64, (3,3),padding="same")(cnn_in)
  cnn_in = BatchNormalization()(cnn_in)
  cnn_in = ReLU()(cnn_in)
  cnn_in = Dropout(0.2)(cnn_in)
  cnn_in = Conv2D(32, (3,3),padding="same")(cnn_in)
  cnn_in = BatchNormalization()(cnn_in)
  cnn_in = ReLU()(cnn_in)
  cnn_in = Dropout(0.2)(cnn_in)
  cnn_in = Conv2D(3, (3,3), padding="same")(cnn_in)
  cnn_in = BatchNormalization()(cnn_in)
  cnn_in = ReLU()(cnn_in)
  cnn_in = Dropout(0.2)(cnn_in)
  ##

  #CNN_BLOCK_TARGET
  cnn_tar = Conv2D(128, (5,5),padding="same")(input_tar)
  cnn_tar = BatchNormalization()(cnn_tar)
  cnn_tar = ReLU()(cnn_tar)
  cnn_tar = Dropout(0.2)(cnn_tar)
  cnn_tar = Conv2D(64, (3,3),padding="same")(cnn_tar)
  cnn_tar = BatchNormalization()(cnn_tar)
  cnn_tar = ReLU()(cnn_tar)
  cnn_tar = Dropout(0.2)(cnn_tar)
  cnn_tar = Conv2D(32, (3,3),padding="same")(cnn_tar)
  cnn_tar = BatchNormalization()(cnn_tar)
  cnn_tar = ReLU()(cnn_tar)
  cnn_tar = Dropout(0.2)(cnn_tar)
  cnn_tar = Conv2D(3, (3,3), padding="same")(cnn_tar)
  cnn_tar = BatchNormalization()(cnn_tar)
  cnn_tar = ReLU()(cnn_tar)
  cnn_tar = Dropout(0.2)(cnn_tar)
  ##

  n_patches = (224 // 64) ** 2

  ## First BORT Layer
  x = GetBROLYFirstLayer(cnn_in, cnn_tar, embedding_dim, attention_heads, ff_depth, n_patches) 
  print(x.shape)
  for _ in range(num_layers_encoder):
    x = GetBROLYEncoderLayer(x, embedding_dim, attention_heads, ff_depth)

  print("BORT CREATED!")  
  context_vector = tf.identity(x)
  print(context_vector.shape)

  x = GetDecoderFirstLayer(context_vector, input_decoder, feed_forward_mask, embedding_dim, attention_heads, ff_depth, seq_len)

  for _ in range(num_layers_decoder):
    x = GetDecoderLayer(context_vector, x, feed_forward_mask, embedding_dim, attention_heads, ff_depth, seq_len)
  
  output = Dense(num_instructions, activation='softmax')(x)

  model = Model([input_in, input_tar, input_decoder, feed_forward_mask], output)

  transformer_optimizer = Get_Custom_Adam_Optimizer(embedding_dim)

  model.compile(optimizer= transformer_optimizer, loss= Transformer_Loss_AIAYN, metrics=['accuracy'])
  model.summary()
  return model