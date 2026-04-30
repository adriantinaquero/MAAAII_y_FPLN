from materials.conllu_reader import ConlluReader, Token

def load_dataset(path: str) -> list[Token]:
    reader = ConlluReader()
    with open(path) as file:
        return [reader.conllustr2tree(i, inference=False) for i in file.read().split("\n\n")]

if __name__ == "__main__":
    tree = load_dataset("FPLN/PRÁCTICA 3/materials/en_partut-ud-train_clean.conllu")
    print("\nConverted Tree Structure:")
    for i in tree[0]: print(i)
