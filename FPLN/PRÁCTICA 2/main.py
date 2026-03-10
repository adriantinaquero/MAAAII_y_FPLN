import keras
from keras.models import Sequential
from keras.layers import Embedding
from keras import layers
from keras import ops
from keras.layers import TextVectorization          # para el tokenizer


batch_size = 128
window_size = 5


def tokenize_text():
    with open("FPLN/PRÁCTICA 2/datasets/game_of_thrones.txt") as file:
        text = file.read()
    tokenizer = TextVectorization.Tokenizer()
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])[0]
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1


def create_windows(text: str):
    windows = []
    labels = []
    for i in range(len(text)-4):
        window = windows.append([text[i], text[i+1], text[i+2], text[i+3], text[i+4]])
        labels.append(text[i+2])



capa_embedding = Embedding(input_dim=window_size, output_dim=64, name="capa_embedding")
modelo = Sequential([
    capa_embedding,
    keras.layers.Average,
    keras.layers.Dense(vocab_size, activation="softmax")
    ])







# Para consultar el embedding de una palabra, extraemos los pesos de la capa de Embedding
embeddings = capa_embedding.get_weights()[0]

# Supongamos que queremos la embedding de la palabra con ID 'word_id'
word_id = 42
word_embedding = embeddings[word_id]

