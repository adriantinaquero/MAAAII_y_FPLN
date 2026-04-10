import numpy as np
import random
from keras import layers
from keras.models import Model




def create_skipgram_windows(sequence: list, window_size: int = 5):
    target_words = []
    context_words = []
    labels = []
    center = window_size // 2
    weights = []
    for i in range(2, len(sequence)-2):
        target = sequence[i]
        context = [sequence[i + k] for k in range(-center, center + 1) if k != 0]
        for word in context:
            target_words.append(target)
            context_words.append(word)
            labels.append(1)
            weights.append(75)
            for i in range(4):
                target_words.append(target)
                random_context = random.choice(sequence)
                context_words.append(random_context)
                labels.append(0)
                weights.append(25)
    return np.array(target_words), np.array(context_words), np.array(labels), np.array(weights)


def drop_words(sequence, t: float = 0.001):
    length = len(sequence)
    freq_dict = {}
    for i in sequence:
        freq_dict[i] = 1 + freq_dict.get(i, 0)
    for i in sequence:
        relative_freq = freq_dict[i] / length
        drop_prob = 1 - np.sqrt(t/relative_freq)
        n = random.randint(0, 100)
        if n < (drop_prob*100):
            sequence = np.concatenate([sequence[:i], sequence[(i+1):]])
    return sequence



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

    # usamos la primera capa target del embedding
    initial_weights = model.layers[2].get_weights()[0]

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model, initial_weights


def train_skipgram_model(train_sequences, test_sequences, vocab_size, window_size=5, batch_size=128, epochs=5):
    train_sequences = drop_words(train_sequences)
    # print(len(sequences))
    target, context, labels, weights = create_skipgram_windows(train_sequences, window_size)
    model, initial_weights = create_skipgram_model(vocab_size)
    model.fit(
        [target, context],
        labels,
        sample_weight=weights,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    trained_weights = model.layers[2].get_weights()[0]

    test_target, test_context, test_labels, test_weights = create_skipgram_windows(test_sequences)
    loss, accuracy = model.evaluate([test_target, test_context], test_labels)
    print("TEST ACCURACY: ", accuracy)

    return initial_weights, trained_weights