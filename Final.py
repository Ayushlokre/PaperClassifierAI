import pathway as pw
import io
from PyPDF2 import PdfReader
import logging

# Reduce Pathway logging verbosity
logging.getLogger("pathway").setLevel(logging.WARNING)

SERVICE_ACCOUNT_FILE = 'file.json'
FOLDER_ID = 'folder_id'

# Step 1: Read data from Google Drive folder using Pathway
# table = pw.io.gdrive.read(
#     object_id=FOLDER_ID,
#     mode="static",  # 'static' mode for one-time reading
#     service_user_credentials_file=SERVICE_ACCOUNT_FILE,
#     with_metadata=True
# )

# Step 2: Define a function to extract text from PDF files
# def extract_text_from_pdf(pdf_bytes):
#     pdf_file = io.BytesIO(pdf_bytes)
#     reader = PdfReader(pdf_file)
#     text = "".join(page.extract_text() or "" for page in reader.pages)
#     return text

# Step 3: Create a new table with the extracted text
# table_with_text = table.select(
#     pdf_text=pw.apply(extract_text_from_pdf, table.data),
#     metadata=table._metadata
# )

# Step 4: Process and display the results using Pathway
# pw.debug.compute_and_print(table_with_text)  # Display the table content
# pw.run()  # Ensure the computation graph runs

import os
import PyPDF2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import spacy
import pandas as pd
import re
from collections import Counter

class PaperClassifier:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        # Added regularization parameters to RandomForestClassifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,  # Limit tree depth
            min_samples_split=5,  # Minimum samples required to split a node
            min_samples_leaf=2,  # Minimum samples in a leaf
            class_weight='balanced',
            random_state=42
        )
        self.category_classifier = None

    def extract_pdf_features(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages[:3]:
                    text += page.extract_text() or ""
                num_pages = len(reader.pages)
                return {'text': text, 'num_pages': num_pages}
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return None

    def compute_linguistic_features(self, text):
        doc = self.nlp(text)
        num_sentences = len(list(doc.sents))
        avg_sentence_length = len(doc) / num_sentences if num_sentences > 0 else 0
        num_entities = len(doc.ents)
        citation_pattern = r'\[\d+\]|\(\w+\s+et\s+al\.\s*,\s*\d{4}\)'
        num_citations = len(re.findall(citation_pattern, text))
        return [avg_sentence_length, num_entities, num_citations, num_sentences]

    def encode_text_with_transformer(self, texts):
        encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def prepare_features(self, pdf_paths):
        texts = []
        numerical_features = []
        for path in pdf_paths:
            features = self.extract_pdf_features(path)
            if features:
                texts.append(features['text'])
                ling_features = self.compute_linguistic_features(features['text'])
                numerical_features.append([features['num_pages']] + ling_features)
        text_features = self.encode_text_with_transformer(texts)
        numerical_features = np.array(numerical_features)
        combined_features = np.hstack([text_features, numerical_features])
        return combined_features

    def train(self, labeled_paths, labels):
        X = self.prepare_features(labeled_paths)
        print(f"Original dataset shape: {Counter(labels)}")
        smote_k_neighbors = min(6, min(Counter(labels).values()) - 1)
        smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, labels)
        self.classifier.fit(X_resampled, y_resampled)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.classifier, X_resampled, y_resampled, cv=skf, scoring='f1')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Average F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    def train_category_classifier(self, labeled_paths, categories):
        X = self.prepare_features(labeled_paths)
        self.category_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,  # Consistent regularization
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.category_classifier.fit(X, categories)

    def predict(self, unlabeled_paths):
        X = self.prepare_features(unlabeled_paths)
        predictions = self.classifier.predict(X)
        categories = []
        for i, pred in enumerate(predictions):
            if pred == 1:
                category = self.category_classifier.predict([X[i]])[0]
                categories.append(category)
            else:
                categories.append("NA")
        return predictions, categories

    def generate_rationale(self, prediction):
        if prediction == 1:
            return "Paper is publishable based on the classification model."
        else:
            return "NA"

def main():
    base_path = "KDSH_2025_Dataset"
    reference_path = os.path.join(base_path, "Reference")
    papers_path = os.path.join(base_path, "Papers")
    labeled_paths = []
    labels = []
    category_labels = []
    publishable_path = os.path.join(reference_path, "Publishable")
    
    # Collecting labeled data
    for conf in ['CVPR', 'EMNLP', 'KDD', 'NeurIPS', 'TMLR']:
        conf_path = os.path.join(publishable_path, conf)
        for paper in os.listdir(conf_path):
            if paper.endswith('.pdf'):
                labeled_paths.append(os.path.join(conf_path, paper))
                labels.append(1)
                category_labels.append(conf)
    
    non_publishable_path = os.path.join(reference_path, "Non-Publishable")
    for paper in os.listdir(non_publishable_path):
        if paper.endswith('.pdf'):
            labeled_paths.append(os.path.join(non_publishable_path, paper))
            labels.append(0)
    
    unlabeled_paths = [
        os.path.join(papers_path, paper)
        for paper in os.listdir(papers_path)
        if paper.endswith('.pdf')
    ]
    
    classifier = PaperClassifier()
    classifier.train(labeled_paths, labels)
    classifier.train_category_classifier(labeled_paths[:len(category_labels)], category_labels)
    predictions, categories = classifier.predict(unlabeled_paths)
    
    # Preparing output in the required format
    results = []
    for i, path in enumerate(unlabeled_paths):
        paper_id = os.path.basename(path)
        publishable = 1 if predictions[i] == 1 else 0
        category = categories[i] if predictions[i] == 1 else "NA"
        rationale = f"Paper is publishable due to its relevance to {category} and high quality based on linguistic and transformer-based features." if predictions[i] == 1 else "NA"
        results.append([paper_id, publishable, category, rationale])
    
    # Saving results to a CSV file
    results_df = pd.DataFrame(results, columns=['Paper ID', 'Publishable', 'Conference', 'Rationale'])
    results_df.to_csv('results.csv', index=False)
    
    # Printing results
    print("\nResults saved to results.csv")
    print("\nSample of predictions:")
    print(results_df.head())
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    y_true = labels
    y_pred = [classifier.classifier.predict([x])[0] for x in classifier.prepare_features(labeled_paths)]
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()

