import os
import pickle
import argparse
import logging

import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


#Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference using trained sentiment model"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (.pkl)"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to inference CSV file"
    )

    return parser.parse_args()


#Main inference pipeline
def main():
    args = parse_args()


    # LOAD MODEL

    try:
        with open(args.model_path, "rb") as f:
            artifacts = pickle.load(f)
            model = artifacts["model"]
            vectorizer = artifacts["vectorizer"]
        logging.info("Model and vectorizer loaded successfully")
    except FileNotFoundError:
        logging.error("Model file not found.")
        raise


    #LOAD DATA

    try:
        df = pd.read_csv(args.data_path)
        logging.info(f"Inference data loaded with shape {df.shape}")
    except FileNotFoundError:
        logging.error("Inference data file not found.")
        raise

    #PREPROCESS DATA
  
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    y_true = df["sentiment"].map({"positive": 1, "negative": 0})
    X = df["review"]

    X_tfidf = vectorizer.transform(X)
    logging.info("Text transformed using trained TF-IDF vectorizer")


    #RUN PREDICTIONS

    y_pred = model.predict(X_tfidf)

  
    #EVALUATION

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Negative", "Positive"]
    )

    logging.info(f"Inference accuracy: {acc:.4f}")


    #OUTPUTS
    os.makedirs("outputs/inference/predictions", exist_ok=True)
    os.makedirs("outputs/inference/figures", exist_ok=True)

    #Save predictions
    predictions_df = pd.DataFrame({
        "review": X,
        "true_label": y_true,
        "predicted_label": y_pred
    })

    predictions_path = "outputs/inference/predictions/predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    logging.info(f"Predictions saved to {predictions_path}")

    #Save metrics
    metrics_path = "outputs/inference/predictions/metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    logging.info(f"Metrics saved to {metrics_path}")


    #CONFUSION MATRIX

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["True Negative", "True Positive"]
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix - Inference")

    cm_path = "outputs/inference/figures/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    logging.info(f"Confusion matrix saved to {cm_path}")
    logging.info("Inference completed successfully")


if __name__ == "__main__":
    main()
