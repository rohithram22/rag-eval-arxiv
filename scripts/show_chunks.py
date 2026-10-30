"""Print the FULL retrieved chunks for one question of an evaluation run.

Usage:
    python scripts/show_chunks.py c076          # all 5 chunks
    python scripts/show_chunks.py c076 4        # only chunk [4]
    python scripts/show_chunks.py c076 4 --run c256-dense
"""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("qid")
parser.add_argument("ranks", nargs="*", type=int, help="chunk numbers to show (default: all)")
parser.add_argument("--run", default="c512-dense")
args = parser.parse_args()

run = f"results/{args.run}"
key = json.load(open(f"{run}/summary.json"))["index_key"]
row = next(r for r in map(json.loads, open(f"{run}/per_question.jsonl")) if r["id"] == args.qid)
rank = {x["id"]: x["rank"] for x in row["retrieved"]}
wanted = set(args.ranks) or set(rank.values())
chunks = [r for r in map(json.loads, open(f"indexes/{key}/nodes.jsonl")) if rank.get(r["id"]) in wanted]
for r in sorted(chunks, key=lambda r: rank[r["id"]]):
    print(f"--- [{rank[r['id']]}] {r['id']} ---\n{r['text']}\n")
