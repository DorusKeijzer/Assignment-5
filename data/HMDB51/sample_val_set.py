import csv
from random import shuffle
def get_csv_row_count(file_path: str) -> int:
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        row_count = sum(1 for row in csv_reader)
    return row_count

rows: int = get_csv_row_count("train_annotation.csv")

val_size: int = int(rows / 10)

indices: list[int] = list(range(rows))
shuffle(indices)

val_indices = indices[:val_size]
train_indices = indices[val_size:]

print(len(val_indices))
print(len(train_indices))

with open("train_annotation.csv") as orig_train:
    with open("val_annotation.csv", "w", newline="") as vals,  open("new_train_annotation.csv", "w", newline="") as new_train:
        valwriter = csv.writer(vals)
        trainwriter = csv.writer(new_train)
        for i, line in enumerate(orig_train):
            file, label = line.split(",")
            label = label.strip()
            if i in val_indices:
                valwriter.writerow([file, label])
            else:
                trainwriter.writerow([file, label])