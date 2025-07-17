import os
import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef


def train(config, train_loader, val_loader, model_save_name, device, RGB_HSI='HSI'):
    model_names = config["model"]["names"]
    if RGB_HSI == 'RGB' :
        in_chans = 3
    elif RGB_HSI == 'HSI':
        in_chans = config["model"]["in_chans"]

    num_classes = config["model"]["num_classes"]

    save_dir = config["model"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    for model_name in model_names:
        print(f"\n Training model: {model_name}")
        model = create_model(model_name=model_name, in_chans=in_chans, num_classes=num_classes, pretrained=True)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
        loss_fn = nn.BCEWithLogitsLoss().to(device)

        best_val_acc = 0.0
        best_model_wts = None

        for epoch in range(config["training"]["epochs"]):
            print(f"[{model_name}] Epoch {epoch+1}/{config['training']['epochs']}")
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            all_probs = []

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).float()

                optimizer.zero_grad()
                outputs = model(images).squeeze(1)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

            epoch_loss = running_loss / len(train_loader.dataset)
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='binary')
            mcc = matthews_corrcoef(all_labels, all_preds)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except:
                auc = float('nan')

            print(f"[{model_name}] Train Loss: {epoch_loss:.4f} | Acc: {acc:.2%} | F1: {f1:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f}")

            model.eval()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad(): 
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).float()
                    outputs = model(images).squeeze(1)
                    loss = loss_fn(outputs, labels)
                    running_loss += loss.item() * images.size(0)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.detach().cpu().numpy())

            epoch_loss = running_loss / len(val_loader.dataset) if loss_fn else None
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='binary')
            mcc = matthews_corrcoef(all_labels, all_preds)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except:
                auc = float('nan')

            
            print(f"[{model_name}] Val Loss: {epoch_loss:.4f} | Acc: {acc:.2%} | F1: {f1:.4f} | AUC: {auc:.4f} | MCC: {mcc:.4f}")

            if acc > best_val_acc:
                best_val_acc = acc
                best_model_wts = model.state_dict() if not hasattr(model, "module") else model.module.state_dict()
                if RGB_HSI == 'RGB':
                    best_model_path = os.path.join(save_dir, f"{'RGB_' + model_name + '_' + model_save_name}_best.pth")
                elif RGB_HSI == 'HSI' :
                    best_model_path = os.path.join(save_dir, f"{'HSI_' + model_name + '_' + model_save_name}_best.pth")
                torch.save(best_model_wts, best_model_path)
                print(f"[{model_name}] New best model saved at: {best_model_path}")
        
        last_model_wts = model.state_dict() if not hasattr(model, "module") else model.module.state_dict()
        if RGB_HSI == 'RGB' :
            last_model_path = os.path.join(save_dir, f"{'RGB_' + model_name + '_' + model_save_name}_last.pth")
        elif RGB_HSI == 'HSI' : 
            last_model_path = os.path.join(save_dir, f"{'HSI_' + model_name + '_' + model_save_name}_last.pth")
        torch.save(last_model_wts, last_model_path)
        print(f"[{model_name}] Last model saved at: {last_model_path}")