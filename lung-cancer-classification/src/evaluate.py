
import torch
import torch.nn as nn
from src.dataset import get_dataloaders
from src.model import DualTransferLungClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

def test_model(data_dir, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_dataloaders(data_dir, batch_size=32)

    model = DualTransferLungClassifier(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Adenocarcinoma', 'Squamous Cell Carcinoma', 'Benign']))

    y_true_bin = np.eye(3)[y_true]
    auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr')
    print(f"AUC Score: {auc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to LC25000 dataset')
    parser.add_argument('--model_path', type=str, default='saved_models/best_model.pth', help='Path to saved model')
    args = parser.parse_args()

    test_model(data_dir=args.data_path, model_path=args.model_path)
