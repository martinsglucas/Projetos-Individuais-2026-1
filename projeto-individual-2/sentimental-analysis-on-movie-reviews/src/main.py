from pathlib import Path
from data.ingest import load_imdb
from transformers import pipeline


DATASET_DIR = Path("data/raw/aclImdb")  # or the absolute Downloads path
df = load_imdb(DATASET_DIR, split="test", sample_size=25000)

classifier = pipeline(
    "text-classification",
    model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device = -1,  # force CPU
    truncation = True,
    max_length = 512,
)

results = classifier(df["text"].tolist(), batch_size=8)

label_map = {"POSITIVE": 1, "NEGATIVE": 0} 
preds = [label_map[r["label"]] for r in results]

correct = sum(p == t for p, t in zip(preds, df["label"].tolist()))
print(f"Accuracy: {correct}/{len(df)} = {correct/len(df):.2%}")
print("\nFirst 3 predictions:")
for i in range(20):
    print(f"    true={df['label'][i]}  pred={preds[i]} conf={results[i]['score']:.3f}")
