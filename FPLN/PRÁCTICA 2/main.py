import cbow
import skipgram
import dataset
import visualize_embeddings



if __name__=="__main__":

    file1 = "FPLN/PRÁCTICA 2/datasets/game_of_thrones.txt"
    file2 = "FPLN/PRÁCTICA 2/materiales/target_words_game_of_thrones.txt"

    train_sequences, test_sequences, tokenizer, vocab_size, word_index = dataset.load_dataset(file1)

    target_words = visualize_embeddings.target_words_list(file2)
    print(target_words)

    embeddings_untrained, embeddings_trained = cbow.train_cbow_model(train_sequences, test_sequences, vocab_size, epochs=5)

    print("Generando visualización: CBOW antes del entrenamiento...")
    visualize_embeddings.visualize_tsne_embeddings(target_words, embeddings_untrained, word_index)

    print("Generando visualización: CBOW después del entrenamiento...")
    visualize_embeddings.visualize_tsne_embeddings(target_words, embeddings_trained, word_index)


    embeddings_untrained, embeddings_trained = skipgram.train_skipgram_model(train_sequences, test_sequences, vocab_size, word_index, target_words, epochs=5)

    print("Generando visualización: Skipgram antes del entrenamiento...")
    visualize_embeddings.visualize_tsne_embeddings(target_words, embeddings_untrained, word_index)

    print("Generando visualización: Skipgram después del entrenamiento...")
    visualize_embeddings.visualize_tsne_embeddings(target_words, embeddings_trained, word_index)