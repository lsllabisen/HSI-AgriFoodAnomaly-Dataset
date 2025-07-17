import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef



def patch_grid(cube, n_lines=100, patch_size=100, stride_lines=100, stride_cols=100, full_width=False):
    H, W, _ = cube.shape

    if full_width:
        patch_size = W
        stride_cols = W

    n_h = (H - n_lines) // stride_lines + 1
    n_w = (W - patch_size) // stride_cols + 1

    return n_h, n_w

def inference_online(model, patches, n_h, n_w, device='cpu'):
    model.eval()
    model.to(device)

    patches = patches.to(device)

    with torch.no_grad():
        outputs = model(patches)
        if outputs.dim() > 2:
            outputs = outputs.squeeze(1)

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()

        try:
            prob_map = probs.view(n_h, n_w).detach().cpu().numpy().tolist()
            pred_map = preds.view(n_h, n_w).detach().cpu().numpy().tolist()
        except RuntimeError as e:
            raise RuntimeError(f"Reshape failed: probs shape={probs.shape}, expected=({n_h},{n_w})") from e

    return preds.cpu().tolist(), probs.cpu().tolist(), pred_map, prob_map 




def inference(model, patches, labels=None, device='cuda'):
    """
    Effectue l'inférence sur un batch de patches avec ou sans labels.

    Args:
        model (torch.nn.Module): Le modèle pré-entraîné.
        patches (torch.Tensor): Tenseur de forme (N, C, H, W).
        labels (List[int] ou torch.Tensor, optional): Liste des labels binaires (0 ou 1).
        device (str): 'cuda' ou 'cpu'.

    Returns:
        preds (List[int]): Prédictions binaires.
        probs (List[float]): Probabilités sigmoidées.
        metrics (dict): (Si labels sont fournis) Dictionnaire de métriques.
    """
    model.eval()
    model.to(device)
    patches = patches.to(device)

    with torch.no_grad():
        outputs = model(patches).squeeze(1)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()

    preds = preds.cpu().tolist()
    probs = probs.cpu().tolist()

    # Si labels disponibles, calculer les métriques
    metrics = {}
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().tolist()

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='binary')
        mcc = matthews_corrcoef(labels, preds)
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = float('nan')

        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "mcc": mcc,
            "auc": auc
        }

    return preds, probs, metrics
