import csv

with open('data/train.csv') as csv_file:
    read_csv = csv.reader(csv_file, delimiter=',')

