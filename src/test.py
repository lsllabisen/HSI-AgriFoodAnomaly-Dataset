import os
import pandas as pd
import torch
import torch.nn as nn
from timm import create_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

def test(config, test_loader, model_save_name, device, checkpoint_type="best", RGB_HSI='HSI'):

    model_names = config["model"]["names"]
    if RGB_HSI == 'RGB' :
        in_chans = 3
    elif RGB_HSI == 'HSI':
        in_chans = config["model"]["in_chans"]
    num_classes = config["model"]["num_classes"]
    save_dir = config["model"]["save_dir"]
    save_results = config["results"]["save_dir"]

    os.makedirs(save_results, exist_ok=True)

    results = []

    for model_name in model_names:
        print(f"\nTesting model: {model_name} ({checkpoint_type})")

        # Initialiser le modèle
        model = create_model(model_name=model_name, in_chans=in_chans, num_classes=num_classes, pretrained=False)
        model = model.to(device)

        # Définir le chemin du checkpoint
        model_path = os.path.join(save_dir, f"{RGB_HSI}_{model_name}_{model_save_name}_{checkpoint_type}.pth")

        print(model_path)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        loss_fn = nn.BCEWithLogitsLoss()

        running_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze(1)
                loss = loss_fn(outputs, labels)
                running_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = running_loss / len(test_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')
        mcc = matthews_corrcoef(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = float('nan')

        print(f"[{model_name}] Test Loss: {test_loss:.4f} | Acc: {acc:.2%} | F1: {f1:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f}")

        results.append({
            "model": model_name,
            "checkpoint": checkpoint_type,
            "loss": test_loss,
            "accuracy": acc,
            "f1_score": f1,
            "auc": auc,
            "mcc": mcc
        })

    # Sauvegarde CSV
    df_results = pd.DataFrame(results)
    results_csv_path = os.path.join(save_results, f"{RGB_HSI}_{model_save_name}_{checkpoint_type}_test_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nTest results saved to: {results_csv_path}")