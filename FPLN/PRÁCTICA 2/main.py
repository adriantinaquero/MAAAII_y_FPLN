import cbow
import skipgram
import dataset
import evaluate



if __name__=="__main__":

    # CARGAMOS DATASET
    txt_file = "FPLN/PRÁCTICA 2/datasets/game_of_thrones.txt"
    target_text_file = "FPLN/PRÁCTICA 2/materiales/target_words_game_of_thrones.txt"
    
    train_sequences, test_sequences, tokenizer, vocabulary, vocab_size, word_index = dataset.load_dataset(txt_file, train_size=0.7)
    target_words = evaluate.target_words_list(target_text_file) # target words para luego visualizar el embedding


    # CBOW
    # creamos y entrenamos el modelo
    embeddings_untrained, embeddings_trained = cbow.train_cbow_model(train_sequences, test_sequences, vocab_size, window_size=5, batch_size=128, epochs=15)

    # visualizamos los embeddings antes y después del entrenamiento
    evaluate.visualize_tsne_embeddings(target_words, embeddings_untrained, word_index, "Embeddings CBOW antes del entrenamiento")
    evaluate.visualize_tsne_embeddings(target_words, embeddings_trained, word_index, "Embeddings CBOW después del entrenamiento")

    # calculamos similitud del coseno y visualizamos 10 palabras más cercanas a una target word aleatoria
    nearest_words = evaluate.nearest_words(target_words, vocabulary, embeddings_trained, word_index)
    evaluate.visualize_nearest_words(target_words, nearest_words, embeddings_trained, word_index)




    # SKIPGRAM
    # creamos y entrenamos el modelo
    embeddings_untrained, embeddings_trained = skipgram.train_skipgram_model(train_sequences, test_sequences, vocab_size, word_index, target_words, window_size=5, batch_size=128, epochs=5)

    # visualizamos los embeddings antes y después del entrenamiento
    evaluate.visualize_tsne_embeddings(target_words, embeddings_untrained, word_index, "Embeddings Skipgram antes del entrenamiento")
    evaluate.visualize_tsne_embeddings(target_words, embeddings_trained, word_index, "Embeddings Skipgram después del entrenamiento")

    # calculamos similitud del coseno y visualizamos 10 palabras más cercanas a una target word aleatoria
    nearest_words = evaluate.nearest_words(target_words, vocabulary, embeddings_trained, word_index)
    evaluate.visualize_nearest_words(target_words, nearest_words, embeddings_trained, word_index)