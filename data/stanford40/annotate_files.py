from glob import glob
import csv
import os
annotation_dict = {"applauding" : 3,
"climbing" : 0,
"drinking" : 9,
"jumping" : 1,
"pouring liquid" : 6,
"riding a bike" : 10,
"riding a horse" : 7,
"running" : 11,
"shooting an arrow" : 4,
"smoking" : 5,
"throwing frisbee" : 8,
"waving hands" : 2}

frequencydict = {}
splits  = glob("ImageSplits/*")

with open("test_annotation.csv", "w", newline="") as test, open ("train_annotation.csv", "w", newline= "") as train:
     testwriter = csv.writer(test)
     trainwriter = csv.writer(train)
     for split in splits:
          label = " ".join(split.split("_")[:-1])[12:]
          set = split.split("_")[-1][:-4]
          print(set)
          writer = testwriter if set == "test" else trainwriter
          annotation = annotation_dict[label]
          with open(split) as splitfile:
               for line in splitfile:
                    writer.writerow([line.split()[0],annotation] )




