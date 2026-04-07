from keras.layers import TextVectorization          # para versiones de keras >= 3



# para versiones de keras >= 3
def tokenize_text(filename: str):
    with open(f"FPLN/PRÁCTICA 2/datasets/{filename}.txt", encoding="utf-8") as file:
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


def load_dataset(filename: str, train_size=0.7):
    sequences, tokenizer, vocab_size = tokenize_text(filename)    
    split_index = int(len(sequences) * train_size)
    train_sequences = sequences[:split_index].tolist()
    test_sequences = sequences[split_index:].tolist()

    return train_sequences, test_sequences, tokenizer, vocab_size