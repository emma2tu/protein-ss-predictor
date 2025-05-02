TRAIN_TSV = "data/train.tsv"
TEST_TSV = "data/test.tsv"
FASTA = "data/sequences.fasta"
PREDICTION_TSV = "data/prediction.tsv"

ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
CLASSIFIER_PATH = "model/rf_model.pkl"
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

residue_13_dict = {"A": "ALA", "R" : "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
residue_31_dict = {"ALA": "A", "ARG" : "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", 	"VAL": "V"}