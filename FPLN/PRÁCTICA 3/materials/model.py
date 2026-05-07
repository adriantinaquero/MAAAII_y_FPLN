from token import Token
import numpy as np
import keras
from algorithm import Sample, ArcEager, Transition


class ParserMLP:
    """
    A Multi-Layer Perceptron (MLP) class for a dependency parser, using TensorFlow and Keras.

    This class implements a neural network model designed to predict transitions in a dependency 
    parser. It utilizes the Keras Functional API, which is more suited for multi-task learning scenarios 
    like this one. The network is trained to map parsing states to transition actions, facilitating 
    the parsing process in natural language processing tasks.

    Attributes:
        word_emb_dim (int): Dimensionality of the word embeddings. Defaults to 100.
        hidden_dim (int): Dimension of the hidden layer in the neural network. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 1.
        batch_size (int): Size of the batches used in training. Defaults to 64.

    Methods:
        train(training_samples, dev_samples): Trains the MLP model using the provided training and 
            development samples. It maps these samples to IDs that can be processed by an embedding 
            layer and then calls the Keras compile and fit functions.

        evaluate(samples): Evaluates the performance of the model on a given set of samples. The 
            method aims to assess the accuracy in predicting both the transition and dependency types, 
            with expected accuracies ranging between 75% and 85%.

        run(sents): Processes a list of sentences (tokens) using the trained model to perform dependency 
            parsing. This method implements the vertical processing of sentences to predict parser 
            transitions for each token.

        Feel free to add other parameters and functions you might need to create your model
    """

    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 64, 
                 epochs: int = 1, batch_size: int = 64):
        """
        Initializes the ParserMLP class with the specified dimensions and training parameters.

        Parameters:
            word_emb_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The size of the hidden layer in the MLP.
            epochs (int): The number of epochs for training the model.
            batch_size (int): The batch size used during model training.
        """

        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size


    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        """
        Trains the MLP model using the provided training and development samples.

        This method prepares the training data by mapping samples to IDs suitable for 
        embedding layers and then proceeds to compile and fit the Keras model.

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
            dev_samples (list[Sample]): A list of development samples used for model validation.
        """

        # creamos vocabulario
        words = set()
        pos_tags = set()
        actions = set()
        dependencies = set()

        for sample in training_samples:
            feats = sample.state_to_feats()
            n = len(feats) // 2         # las features son de tipo [words,..., pos,...]
            feat_words = feats[:n]
            feat_pos = feats[n:]

            words.update(feat_words)
            pos_tags.update(feat_pos)
            actions.add(sample.transition.action)
            if sample.transition.dependency is not None:
                dependencies.add(sample.transition.dependency)

        # añadimos PAD y UNK. También añadimos NONE para las dependencies de SHIFT y REDUCE
        words.update(["<PAD>", "<UNK>"])
        pos_tags.update(["<PAD>", "<UNK>"])
        dependencies.add("<NONE>")

        # creamos diccionarios para mapear a IDs
        self.word_to_id = {}
        for i, word in enumerate(sorted(words)):
            self.word_to_id[word] = i

        self.pos_to_id = {}
        for i, pos in enumerate(sorted(pos_tags)):
            self.pos_to_id[pos] = i
        
        self.action_to_id = {}
        for i, action in enumerate(sorted(actions)):
            self.action_to_id[action] = i        
        
        self.dependency_to_id = {}
        for i, dependency in enumerate(sorted(dependencies)):
            self.dependency_to_id[dependency] = i        

        self.id_to_action = {}
        for action in self.action_to_id:
            index = self.action_to_id[action]
            self.id_to_action[index] = action        
    
        self.id_to_dependency = {}
        for dependency in self.dependency_to_id:
            index = self.dependency_to_id[dependency]
            self.id_to_dependency[index] = dependency 

        # una vez creados los diccionarios, mapeamos las palabras
        X_words = []
        X_pos = []
        y_action = []
        y_dependency = []

        for sample in training_samples:
            feats = sample.state_to_feats()
            n = len(feats) // 2
            feat_words = feats[:n]
            feat_pos = feats[n:]

            word_ids = []
            for word in feat_words:
                if word in self.word_to_id:
                    word_ids.append(self.word_to_id[word])
                else:
                    word_ids.append(self.word_to_id["<UNK>"])

            pos_ids = []
            for p in feat_pos:
                if p in self.pos_to_id:
                    pos_ids.append(self.pos_to_id[p])
                else:
                    pos_ids.append(self.pos_to_id["<UNK>"])

            X_words.append(word_ids)
            X_pos.append(pos_ids)

            y_action.append(self.action_to_id[sample.transition.action])

            dependency = sample.transition.dependency
            if dependency is None:
                dependency = "<NONE>"
            y_dependency.append(self.dependency_to_id[dependency])

            X_words = np.array(X_words)
            X_pos = np.array(X_pos)
            y_action = np.array(y_action)
            y_dependency = np.array(y_dependency)

            # creamos el modelo
            n_word_feats = X_words.shape[1]       # miramos el número de features por palabra para meterlo como tamaño del input del modelo
            n_pos_feats = X_pos.shape[1]

            word_input = keras.layers.Input(shape=(n_word_feats,))
            pos_input = keras.layers.Input(shape=(n_pos_feats,))

            word_emb = keras.layers.Embedding(input_dim=len(self.word_to_id), output_dim=self.word_emb_dim)(word_input)
            pos_emb = keras.layers.Embedding(input_dim=len(self.pos_to_id), output_dim=32)(pos_input)        # pusimos 32 como pos_embedding_size

            word_flat = keras.layers.Flatten()(word_emb)
            pos_flat = keras.layers.Flatten()(pos_emb)

            concatenation = keras.layers.Concatenate()([word_flat, pos_flat])

            hidden = keras.layers.Dense(self.hidden_dim, activation="relu")(concatenation)

            action_output = keras.layers.Dense(len(self.action_to_id), activation="softmax", name="action_output")(hidden)
            dependency_output = keras.layers.Dense(len(self.dependency_to_id), activation="softmax", name="dependency_output")(hidden)

            self.model = keras.Model(inputs=[word_input, pos_input], outputs=[action_output, dependency_output])

            # compilamos y entrenamos el modelo
            self.model.compile(
                optimizer="adam", 
                loss={"action_output": "categorical_crossentropy", "dependency_output": "categorical_crossentropy"},
                metrics={"action_output": "accuracy", "dependency_output": "accuracy"}
            )

            self.model.fit(
                [X_words, X_pos],
                {"action_output": y_action, "dependency_output": y_dependency},
                epochs=self.epochs,
                batch_size=self.batch_size,
            )

    def evaluate(self, samples: list['Sample']):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            samples (list[Sample]): A list of samples to evaluate the model's performance.
        """

        # traducimos los samples a sus IDs
        X_words = []
        X_pos = []
        y_action = []
        y_dependency = []

        for sample in samples:

            feats = sample.state_to_feats()
            n = len(feats) // 2
            feat_words = feats[:n]
            feat_pos = feats[n:]

            word_ids = []
            for w in feat_words:
                if w in self.word_to_id:
                    word_ids.append(self.word_to_id[w])
                else:
                    word_ids.append(self.word_to_id["<UNK>"])

            pos_ids = []
            for p in feat_pos:
                if p in self.pos_to_id:
                    pos_ids.append(self.pos_to_id[p])
                else:
                    pos_ids.append(self.pos_to_id["<UNK>"])

            X_words.append(word_ids)
            X_pos.append(pos_ids)

            y_action.append(self.action_to_id[sample.transition.action])

            dependency = sample.transition.dependency
            if dependency is None:
                dependency = "<NONE>"
            y_dependency.append(self.dependency_to_id[dependency])

        X_words = np.array(X_words)
        X_pos = np.array(X_pos)
        y_action = np.array(y_action)
        y_dependency = np.array(y_dependency)

        # evaluación
        results = self.model.evaluate(
            [X_words, X_pos],
            {"action_output": y_action, "dependency_output": y_dependency},
            verbose=1
        )

        # el modelo devuelve
        # results[0] -> total loss
        # results[1] -> action loss
        # results[2] -> dependency loss
        # results[3] -> action accuracy
        # results[4] -> dependency accuracy

        # mostramos resultados
        print("\nEvaluation results:")
        print(f"Total loss: {results[0]:.4f}")
        print(f"Action accuracy: {results[3]:.4f}")
        print(f"Dependency accuracy: {results[4]:.4f}")


    def run(self, sents: list['Token']):
        """
        Executes the model on a list of sentences to perform dependency parsing.

        This method implements the vertical processing of sentences, predicting parser 
        transitions for each token in the sentences.

        Parameters:
            sents (list[Token]): A list of sentences, where each sentence is represented 
                                    as a list of Token objects.
        """

        # Main Steps for Processing Sentences:
        # 1. Initialize: Create the initial state for each sentence.
        # 2. Feature Representation: Convert states to their corresponding list of features.
        # 3. Model Prediction: Use the model to predict the next transition and dependency type for all current states.
        # 4. Transition Sorting: For each prediction, sort the transitions by likelihood using numpy.argsort, 
        #    and select the most likely dependency type with argmax.
        # 5. Validation Check: Verify if the selected transition is valid for each prediction. If not, select the next most likely one.
        # 6. State Update: Apply the selected actions to update all states, and create a list of new states.
        # 7. Final State Check: Remove sentences that have reached a final state.
        # 8. Iterative Process: Repeat steps 2 to 7 until all sentences have reached their final state.
        
        return NotImplementedError


if __name__ == "__main__":
    
    model = ParserMLP()