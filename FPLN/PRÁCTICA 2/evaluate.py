import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import random


def target_words_list(file: str):
    with open(file, 'r', encoding='utf-8') as file:
        target_words = [line.strip() for line in file]
    return target_words


def nearest_words(target_words, all_words, embeddings, word_index):
    nearest_words = []
    for i in range(len(target_words)):
        word_distances = []
        indice = word_index[target_words[i]]
        word1_embedding = embeddings[indice]
        for j in range(len(all_words)):
            indice = word_index[all_words[j]]
            word2_embedding = embeddings[indice]
            similarity = cosine_similarity([word1_embedding], [word2_embedding])[0][0]
            word_distances.append((all_words[j], similarity))
        word_distances.sort(key=lambda x: x[1], reverse=True)
        top_10 = [word for word, sim in word_distances[:11]]
        nearest_words.append(top_10)
    
    return nearest_words


# Función que elige una target_word aleatoria y muestra la gráfica de sus
def visualize_nearest_words(target_words, nearest_words, embeddings, word_index):
    random_index = random.randrange(0, (len(target_words)-1))
    random_word = target_words[random_index]
    close_words = nearest_words[random_index]
    titulo = "10 palabras más cercanas a " + str(random_word)
    visualize_tsne_embeddings(close_words, embeddings, word_index, titulo)


def visualize_tsne_embeddings(words, embeddings, word_index, titulo, filename=None):
    """
    Visualizes t-SNE embeddings of selected words.

    Args:
        words (list): List of words to visualize.
        embeddings (numpy.ndarray): Array containing word embeddings.
        word_index (dict): Mapping of words to their indices in the embeddings array.
        filename (str, optional): File to save the visualization. If None, plot is displayed.

    Returns:
        None
    """
    # Filter the embeddings for the selected words
    indices = [word_index[word] for word in words]
    selected_embeddings = embeddings[indices]

    # Set perplexity for t-SNE, it's recommended to use a value less than the number of selected words
    perplexity = min(5,len(words) - 1)

    # Use t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.title(titulo, fontsize=15, pad=20)
    for i, word in enumerate(words):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')

    # Save or display the plot
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def visualize_all_tsne_embeddings(words_to_plot, embeddings, word_index, titulo, words_to_label=None, filename=None):
    """
    Visualizes t-SNE embeddings of selected words with optional labeling.

    Args:
        embeddings (numpy.ndarray): Array containing word embeddings.
        word_index (dict): Mapping of words to their indices in the embeddings array.
        words_to_plot (list): List of words to plot.
        words_to_label (list, optional): List of words to label. Defaults to None.
        filename (str, optional): File to save the visualization. If None, plot is displayed.

    Returns:
        None
    """
    # Create a reverse mapping from index to word
    index_word = {index: word for word, index in word_index.items()}

    # Ensure words_to_label is a subset of words_to_plot
    if words_to_label is None:
        words_to_label = words_to_plot
    words_to_label = set(words_to_label).intersection(words_to_plot)

    # Filter the embeddings for the words to plot
    indices_to_plot = [word_index[word] for word in words_to_plot if word in word_index]
    selected_embeddings = embeddings[indices_to_plot]

    # Set perplexity for t-SNE, it's recommended to use a value less than the number of selected words
    perplexity = min(5,len(words_to_plot) - 1)

    # Use t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    # Plotting
    plt.figure(figsize=(12, 12))
    plt.title(titulo, fontsize=15, pad=20)
    for i, index in enumerate(indices_to_plot):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], alpha=0.5)
        if index_word[index] in words_to_label:  # Annotate only selected words
            plt.annotate(index_word[index],
                         xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')