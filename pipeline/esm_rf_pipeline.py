from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
import numpy as np
import joblib
import os
from tqdm import tqdm
from config import ESM_MODEL_NAME, CLASSIFIER_PATH

class ESMRandomForestPipeline:
    def __init__(self, model_name=ESM_MODEL_NAME, classifier_path=CLASSIFIER_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)
        self.classifier_path = classifier_path

        self.embedding_cache_dir = "cache/embeddings/"
        os.makedirs(self.embedding_cache_dir, exist_ok=True)

    def embed_sequence(self, sequence, protein_name=None):
        """Return per-residue embeddings (logits) for a given amino acid sequence."""
        if protein_name:
            cache_file = os.path.join(self.embedding_cache_dir, f"{protein_name}.npy")
            if os.path.exists(cache_file):
                return np.load(cache_file)
        
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Exclude CLS and EOS tokens
        residue_embeddings = outputs.last_hidden_state[0][1:-1]

        result = residue_embeddings.detach().cpu().numpy()  # Safely move back to CPU

        if protein_name:
            np.save(cache_file, result)

        return result

    def generate_inputset(self, inputset_dict):
        X = []
        for protein, seq in inputset_dict.items():
            embeddings = self.embed_sequence(seq, protein_name=protein)
        return np.array(X)

    def generate_dataset(self, dataset_dict):
        """
        Converts a protein → (sequence, labels) dict into training data.

        Returns:
            - X: List of embedding vectors
            - y: Corresponding structure labels
        """
        X, y = [], []
        for i, (protein, (seq, labels)) in enumerate(tqdm(dataset_dict.items(), desc="Embedding training set")):
            embeddings = self.embed_sequence(seq, protein_name=protein) 
            for i, emb in enumerate(embeddings):
                if i < len(labels): #safety check
                    X.append(emb)
                    y.append(labels[i])
        return np.array(X), np.array(y)

    def train(self, training_set):
        X_train, y_train = self.generate_dataset(training_set)
        self.classifier.fit(X_train, y_train)
        self.save_classifier()

    def predict(self, sequence):
        embeddings = self.embed_sequence(sequence)
        predictions = self.classifier.predict(embeddings)
        return predictions

    def evaluate(self, validation_set):
        X_val, y_val = self.generate_dataset(validation_set)
        y_pred = self.classifier.predict(X_val)
        print("Validation Predictions: ", y_pred)
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        print("Validation Accuracy:", acc)
        print(report)
        return acc, report
    
    def test(self, testing_set):
        X_test = self.generate_inputset(testing_set)
        y_pred = self.classifier.predict(X_test)
        return y_pred 
    
    def save_classifier(self):
        os.makedirs(os.path.dirname(self.classifier_path), exist_ok=True)
        joblib.dump(self.classifier, self.classifier_path)

    def load_classifier(self):
        if os.path.exists(self.classifier_path):
            self.model = joblib.load(self.classifier_path)
        else:
            raise FileNotFoundError(f"Model file not found at {self.classifier_path}")
