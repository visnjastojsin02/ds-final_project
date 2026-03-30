import os
import pickle
import argparse
import logging

from matplotlib import cm
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)



# ARGUMENT PARSER

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train sentiment classification model"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training CSV file"
    )

    return parser.parse_args()



# MAIN TRAINING PIPELINE
def main():
    args = parse_args()


    try:
        df = pd.read_csv(args.data_path)
        logging.info(f"Dataset loaded successfully with shape {df.shape}")
    except FileNotFoundError:
        logging.error("Training data file not found.")
        raise


    #LABEL PREPROCESSING

    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    y = df["sentiment"].map({"positive": 1, "negative": 0})
    X = df["review"]

    logging.info("Labels mapped: positive -> 1, negative -> 0")


    # TF-IDF VECTORIZATION
   
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )

    X_tfidf = vectorizer.fit_transform(X)
    logging.info("TF-IDF vectorization completed")


    # TRAIN / VALIDATION SPLIT

    X_train, X_val, y_train, y_val = train_test_split(
        X_tfidf,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logging.info("Train-validation split completed")


    # TRAIN FINAL MODEL

    model = LinearSVC(C=0.5)
    model.fit(X_train, y_train)

    logging.info("Linear SVM model trained")


    # EVALUATION
 
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    logging.info(f"Validation accuracy: {acc:.4f}")
    
    cm = confusion_matrix(y_val, y_pred)
    os.makedirs("outputs/train/figures", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["True Negative", "True Positive"]
    )

    figure_path = "outputs/train/figures/confusion_matrix.png"
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(figure_path)
    plt.close() 
    logging.info(f"Confusion matrix saved to {figure_path}")


    #SAVE OUTPUTS
  
    os.makedirs("outputs/train/models", exist_ok=True)
    os.makedirs("outputs/train/predictions", exist_ok=True)

    # Save model + vectorizer
    model_path = "outputs/train/models/svm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "vectorizer": vectorizer
            },
            f
        )

    logging.info(f"Model saved to {model_path}")

    #Save metrics
    metrics_path = "outputs/train/predictions/metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    logging.info(f"Metrics saved to {metrics_path}")
    logging.info("Training completed successfully")



if __name__ == "__main__":
    main()


