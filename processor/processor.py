import config
import pandas as pd
from Bio import SeqIO
import joblib
import os

class ProteinDataProcessor:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.train_tsv_dict = {}
        self.test_tsv_dict = {}
        self.raw_sequences_dict = {}
        self.training_sequences_dict = {}
        self.testing_sequences_dict = {}
        self.training_set = {}
        self.validation_set = {}
        self.testing_set = {}
        self.cache_path = "cache/processor.pkl"

    def __str__(self):
        summary = (
            "ProteinDataProcessor Summary:\n"
            f"  Proteins in raw FASTA: {len(self.raw_sequences_dict)}\n"
            f"  Proteins in train.tsv: {len(self.train_tsv_dict)}\n"
            f"  Proteins in test.tsv: {len(self.test_tsv_dict)}\n"
            f"  Filtered training sequences: {len(self.training_sequences_dict)}\n"
            f"  Filtered testing sequences: {len(self.testing_sequences_dict)}\n"
            f"  Training set size: {len(self.training_set)}\n"
            f"  Validation set size: {len(self.validation_set)}\n"
            f"  Testing set size: {len(self.testing_set)}"
        )
        return summary

    def process_train_tsv(self, train_tsv):
        self.train_df = pd.read_csv(train_tsv, sep="\t", header=0).dropna()
        for _, row in self.train_df.iterrows():
            protein, _, pos = row['id'].split("_")
            position = int(pos.strip())
            sec_struct = row['secondary_structure'].strip()
            if protein not in self.train_tsv_dict:
                self.train_tsv_dict[protein] = []
            while len(self.train_tsv_dict[protein]) < position:
                self.train_tsv_dict[protein].append(".")
            self.train_tsv_dict[protein][position - 1] = sec_struct
        return self.train_tsv_dict

    def process_test_tsv(self, test_tsv):
        self.test_df = pd.read_csv(test_tsv, sep="\t", header=0).dropna()
        for _, row in self.test_df.iterrows():
            protein = row['id'].split("_")[0].strip()
            residue_id = row['id'].strip()
            self.test_tsv_dict.setdefault(protein, []).append(residue_id)
        return self.test_tsv_dict

    def process_sequences_fasta(self, sequences_fasta):
        for record in SeqIO.parse(sequences_fasta, "fasta"):
            self.raw_sequences_dict[record.id.strip()] = str(record.seq).strip()

    def filter_training_sequences(self):
        for protein, seq in self.raw_sequences_dict.items():
            if protein in self.train_tsv_dict and len(seq) >= len(self.train_tsv_dict[protein]):
                self.training_sequences_dict[protein] = seq
        return self.training_sequences_dict

    def filter_testing_sequences(self):
        for protein, seq in self.raw_sequences_dict.items():
            if protein in self.test_tsv_dict:
                self.testing_sequences_dict[protein] = seq
        return self.testing_sequences_dict

    def make_training_validation_sets(self, val_every=5):
        self.training_set.clear()
        self.validation_set.clear()
        count = 0
        for protein, seq in self.training_sequences_dict.items():
            if any(aa not in config.residue_13_dict for aa in seq):
                continue
            sec_struct = self.train_tsv_dict[protein]
            count += 1
            if count % val_every == 0:
                self.validation_set[protein] = (seq, sec_struct)
            else:
                self.training_set[protein] = (seq, sec_struct)
        return self.training_set, self.validation_set

    def make_testing_set(self):
        for protein, seq in self.testing_sequences_dict.items():
            if any(aa not in config.residue_13_dict for aa in seq):
                continue
            self.testing_set[protein] = seq
        return self.testing_set

    def process_all(self, train_tsv=config.TRAIN_TSV, test_tsv=config.TEST_TSV, sequences_fasta=config.FASTA):
        if os.path.exists(self.cache_path):
            print("Loading cached processor data...")
            cached = joblib.load(self.cache_path)
            self.__dict__.update(cached.__dict__)
            return
        
        print("Processing files from scratch...")
        self.process_train_tsv(train_tsv)
        self.process_test_tsv(test_tsv)
        self.process_sequences_fasta(sequences_fasta)
        self.filter_training_sequences()
        self.filter_testing_sequences()
        self.make_training_validation_sets()
        self.make_testing_set()
        print(f"Processed:")
        print(f"  Training proteins: {len(self.training_set)}")
        print(f"  Validation proteins: {len(self.validation_set)}")
        print(f"  Testing proteins: {len(self.testing_set)}")

        # Save to cache
        os.makedirs("cache", exist_ok=True)
        joblib.dump(self, self.cache_path)
        print("Processor data cached.")
