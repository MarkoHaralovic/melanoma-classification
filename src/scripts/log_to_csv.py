import argparse
import csv
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()

    input_file = os.path.join(args.experiment, "log.txt")
    output_file = os.path.join(args.experiment, "log.csv")

    with open(input_file, "r") as fin, open(output_file, "w", newline="") as fout:
        lines = fin.readlines()

        keys = list(json.loads(lines[0]).keys())
        if "epoch" in keys:
            keys.remove("epoch")

        header = ["epoch"] + keys
        writer = csv.DictWriter(fout, fieldnames=header)
        writer.writeheader()

        for epoch, line in enumerate(lines):
            data = json.loads(line)
            writer.writerow(data)

    print(f"Done, saved to file {output_file}")
