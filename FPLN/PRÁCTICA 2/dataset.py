from keras.layers import TextVectorization          # para versiones de keras >= 3


# para versiones de keras >= 3
def tokenize_text(file: str):
    with open(file, encoding="utf-8") as file:
        text = file.read()

    vectorizer = TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        output_mode="int"
    )
    vectorizer.adapt([text])
    sequence = vectorizer([text]).numpy()[0]
    vocabulary = vectorizer.get_vocabulary()
    vocab_size = len(vocabulary)

    word_index = {word: i for i, word in enumerate(vocabulary)}

    return sequence, vectorizer, vocabulary, vocab_size, word_index


def load_dataset(file: str, train_size=0.7):
    sequences, tokenizer, vocabulary, vocab_size, word_index = tokenize_text(file)    
    split_index = int(len(sequences) * train_size)
    train_sequences = sequences[:split_index].tolist()
    test_sequences = sequences[split_index:].tolist()

    return train_sequences, test_sequences, tokenizer, vocabulary, vocab_size, word_index