from random import shuffle
import random

random.seed(1234)

with open("reviews.txt", "r") as fh:
    reviews = fh.read().split("\n")

shuffle(reviews)

train_data = reviews[:int(0.8* len(reviews))]
test_data = reviews[int(0.8* len(reviews)): ]

with open("train.txt", "w") as fh:
    fh.write("\n".join(train_data))

with open("dev.txt", "w") as fh:
    fh.write("\n".join(test_data))