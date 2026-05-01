import pandas as pd
import json

CSV_PATH = "LLaVA-Med/QAPairs_Finalized.csv"
OUT_PATH = "LLaVA-Med/QAPairs_Finalized.jsonl"

IMAGE_SUFFIX = "_montage.png"  

df = pd.read_csv(CSV_PATH)

records = []
qid = 0

for _, row in df.iterrows():
    if row["Tag"] != "YES":
        continue

    accession = row["Accession Num"]
    image_name = f"{accession}{IMAGE_SUFFIX}"

    record = {
        "question_id": qid,
        "image": image_name,
        "text": row["Question"].strip() + "\n<image>",
        "answer": row["Answer"].strip()
    }

    records.append(record)
    qid += 1

with open(OUT_PATH, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"Wrote {qid} QA pairs to {OUT_PATH}")
