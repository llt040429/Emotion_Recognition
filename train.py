import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


class AudioCsvDataset(Dataset):
    def __init__(self, csv_path, sample_rate=16000, max_seconds=4.0):
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)

    def __len__(self):
        return len(self.df)

    def _fix_length(self, wav):
        if wav.numel() > self.max_samples:
            wav = wav[: self.max_samples]
        elif wav.numel() < self.max_samples:
            wav = torch.nn.functional.pad(wav, (0, self.max_samples - wav.numel()))
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        label = int(row["label_id"])

        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = self._fix_length(wav)
        return {"audio": wav.numpy(), "label": label}


def collate_fn_builder(feature_extractor, sample_rate):
    def collate_fn(batch):
        audios = [b["audio"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        feats = feature_extractor(
            audios,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        feats["labels"] = labels
        return feats
    return collate_fn


@torch.no_grad()
def extract_embeddings(model, loader, device, use_fp16=False):
    model.eval()
    all_embs = []
    all_labels = []

    for batch in tqdm(loader, desc="extract", leave=False):
        labels = batch.pop("labels")
        input_values = batch["input_values"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda" and use_fp16)
        ):
            out = model(input_values=input_values, attention_mask=attention_mask)
            hidden = out.last_hidden_state  # [B, T_feat, D]

            if attention_mask is not None:
                # 将 sample-level mask 映射到 feature-level mask，避免维度不匹配
                feat_mask = model._get_feature_vector_attention_mask(
                    hidden.shape[1], attention_mask
                )  # [B, T_feat]
                mask = feat_mask.unsqueeze(-1).to(hidden.dtype)  # [B, T_feat, 1]
                summed = (hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1e-6)
                emb = summed / lengths
            else:
                emb = hidden.mean(dim=1)

        all_embs.append(emb.detach().cpu())
        all_labels.append(labels)

    all_embs = torch.cat(all_embs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_embs, all_labels


def l2_normalize(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def main():
    parser = argparse.ArgumentParser("No-finetune baseline eval with class centroids")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--labels-csv", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="outputs/baseline_no_finetune_metrics.json")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels_csv)
    id2label = {int(r.label_id): r.label for r in labels_df.itertuples(index=False)}
    n_classes = len(id2label)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = AudioCsvDataset(args.train_csv, args.sample_rate, args.max_seconds)
    test_ds = AudioCsvDataset(args.test_csv, args.sample_rate, args.max_seconds)

    collate_fn = collate_fn_builder(feature_extractor, args.sample_rate)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # 1) 提取训练/测试 embedding
    x_train, y_train = extract_embeddings(model, train_loader, device, args.fp16)
    x_test, y_test = extract_embeddings(model, test_loader, device, args.fp16)

    # 2) 归一化 + 类中心
    x_train = l2_normalize(x_train)
    x_test = l2_normalize(x_test)

    centroids = np.zeros((n_classes, x_train.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        idx = np.where(y_train == c)[0]
        if len(idx) == 0:
            continue
        centroids[c] = x_train[idx].mean(axis=0)
    centroids = l2_normalize(centroids)

    # 3) 余弦相似度最近中心分类
    sims = x_test @ centroids.T  # [N, C]
    y_pred = sims.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    report = classification_report(
        y_test, y_pred,
        target_names=[id2label[i] for i in range(n_classes)],
        digits=4, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "method": "wavlm_no_finetune_class_centroid",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "device": str(device),
        "num_test_samples": int(len(y_test)),
        "classification_report": report,
        "confusion_matrix": cm
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "output_json": str(out_path)
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
