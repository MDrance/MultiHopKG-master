from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from torch import Tensor
import json

def transformer_embeddings(data_path: str, entity2id: dict) -> tuple[Tensor]:
    """
    Match the entity2id dict from KnowledgeGraph class to a mapping {ID: DEFINITION, ...}
    Using this mapping, create embeddings given the DEFINITION using Transformer
    Embeddings will be ordered matching the entity2id and relation2id
    """
    id_to_def_path = data_path + "/id_to_def.json"
    with open(id_to_def_path, "r") as id_to_def_json:
        id_to_def = json.load(id_to_def_json)
    id_to_def["DUMMY_ENTITY"] = ""
    id_to_def["NO_OP_ENTITY"] = ""

    def_vocab = {}
    transformer = SentenceTransformer("all-MiniLM-L6-v2")

    for key, value in entity2id.items():
        def_vocab[value] = id_to_def[key]
    print("Corpus sanity check :")
    print("entity2id KEY : VALUE = {0} : {1}".format( "14854262", entity2id["14854262"]))
    print("id_to_def KEY : VALUE = {0} : {1}".format("14854262", id_to_def["14854262"]))
    print("def_vocab for KEY : {0}".format(def_vocab[entity2id["14854262"]]))
    
    corpus = list(id_to_def.values())
    print("Constructing BERT embeddings")
    corpus_embeddings = transformer.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

    return corpus_embeddings
