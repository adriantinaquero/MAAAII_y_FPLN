import cbow
import skipgram
import dataset



if __name__=="__main__":

    train_sequences, test_sequences, tokenizer, vocab_size = dataset.load_dataset("game_of_thrones")

    cbow.train_cbow_model(train_sequences, test_sequences, vocab_size)

    skipgram.train_skipgram_model(train_sequences, test_sequences, vocab_size)

    # # Para consultar el embedding de una palabra, extraemos los pesos de la capa de Embedding
    # embeddings = model.layers[1].get_weights()[0]

    # # Supongamos que queremos la embedding de la palabra con ID 'word_id'
    # word_id = 42
    # word_embedding = embeddings[word_id]
