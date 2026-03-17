import numpy as np
from keras import layers
from keras.models import Model
from keras.layers import TextVectorization          # para versiones de keras >= 3


# Deberíamos hacer el create_windows para un window_size variable!!


# para versiones de keras >= 3
def tokenize_text():
    with open("FPLN/PRÁCTICA 2/datasets/game_of_thrones.txt", encoding="utf-8") as file:
        text = file.read()

    vectorizer = TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        output_mode="int"
    )
    vectorizer.adapt([text])
    sequence = vectorizer([text]).numpy()[0]
    vocab_size = len(vectorizer.get_vocabulary())

    return sequence, vectorizer, vocab_size


def create_cbow_windows(sequences: str, n: int = 5):
    windows = []
    labels = []
    center = n // 2
    for i in range(len(sequences) - n + 1):
        windows.append([sequences[i + k] for k in range(n) if k != center])
        labels.append(sequences[i + center])
    return np.array(windows), np.array(labels)


def create_cbow_model(vocab_size, embedding_dim=100):
    inputs = layers.Input(shape=(4,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def create_skipgram_windows(sequence: str, n: int = 5):
    target_words = []
    context_words = []
    labels = []
    center = n // 2
    for i in range(2, len(sequence)-2):
        target = sequence[i]
        context = [sequence[i + k] for k in range(-center, center + 1) if k != 0]
        for word in context:
            target_words.append(target)
            context_words.append(word)
            labels.append(1)
    return np.array(target_words), np.array(context_words), np.array(labels)


def create_skipgram_model(vocab_size, embedding_dim=100):
    target_input = layers.Input(shape=(1,))
    context_input = layers.Input(shape=(1,))

    target_layer = layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim)(target_input)
    context_layer = layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim)(context_input)

    target_layer = layers.Flatten()(target_layer)
    context_layer = layers.Flatten()(context_layer)

    dot_product = layers.Dot(axes=1)([target_layer, context_layer])

    output = layers.Dense(1, activation="sigmoid")(dot_product)

    model = Model(inputs=[target_input, context_input], outputs=output)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__=="__main__":

    # enrenamos CBOW
    sequences, tokenizer, vocab_size = tokenize_text()
    context, labels = create_cbow_windows(sequences)
    model = create_cbow_model(vocab_size)
    model.summary()
    model.fit(
        context,
        labels,
        batch_size=128,
        epochs=10
    )


    # entrenamos Skipgram
    sequences, tokenizer, vocab_size = tokenize_text()
    target, context, labels = create_skipgram_windows(sequences)
    model = create_skipgram_model(vocab_size)
    model.fit(
        [target, context],
        labels,
        batch_size=128,
        epochs=10
    )



    # # Para consultar el embedding de una palabra, extraemos los pesos de la capa de Embedding
    # embeddings = model.layers[1].get_weights()[0]

    # # Supongamos que queremos la embedding de la palabra con ID 'word_id'
    # word_id = 42
    # word_embedding = embeddings[word_id]
