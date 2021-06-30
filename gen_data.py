import time
import tensorwatch as tw
import random

# create watcher, notice that we are not logging anything
w = tw.Watcher()

for i in range(10000):
    x = i
    loss = random.random() * 10
    train_accuracy = random.random()

    # we are just observing variables
    # observation has no cost, nothing gets logged anywhere
    w.observe(iter=x, loss=loss, train_accuracy=train_accuracy)

    time.sleep(1)
