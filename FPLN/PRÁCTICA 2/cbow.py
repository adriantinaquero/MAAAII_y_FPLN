import numpy as np
from keras import layers
from keras.models import Model
import visualize_embeddings



def create_cbow_windows(sequences: list, n: int = 5):
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

    # usamos la primera capa target del embedding
    initial_weights = model.layers[1].get_weights()[0]

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, initial_weights


def train_cbow_model(train_sequences, test_sequences, vocab_size, batch_size=128, epochs=10):

    context, labels = create_cbow_windows(train_sequences)
    model, initial_weights = create_cbow_model(vocab_size)
    model.summary()
    model.fit(
        context,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    trained_weights = model.layers[1].get_weights()[0]

    test_context, test_labels = create_cbow_windows(test_sequences)
    loss, accuracy = model.evaluate(test_context, test_labels)
    print("TEST ACCURACY: ", accuracy)

    return initial_weights, trained_weights