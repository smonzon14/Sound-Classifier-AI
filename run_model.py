# Author: Sebastian Monzon

from multiclass import MulticlassNSynth
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import loader

INSTRUMENT_FAMILIES = ["bass", "brass", "flute", "guitar", "keyboard",
                       "mallet", "organ", "reed", "string", "synth_lead", "vocal"]

BATCH_SIZE = 64
dataset_test = loader.nsynth_dataset(batch_size=BATCH_SIZE, split="test")

def confusion(y_test,pred):
    plt.figure(figsize=(10,6))
    fx = sns.heatmap(confusion_matrix(y_test,pred), annot=True, fmt=".1f",cmap="GnBu")
    fx.set_title('Confusion Matrix \n')
    fx.set_xlabel('\n Predicted Values\n')
    fx.set_ylabel('Actual Values\n')
    fx.xaxis.set_ticklabels(INSTRUMENT_FAMILIES)
    fx.yaxis.set_ticklabels(INSTRUMENT_FAMILIES)
    plt.yticks(rotation=0)
    plt.xticks(rotation=-30)

    plt.show()

# load model
model = MulticlassNSynth(batch_size=BATCH_SIZE)
model.load_weights("model_checkpoint")

pred = []
y_test = []

for batch in dataset_test:
    x, labels = batch
    for l in labels:
        y_test.append(l)
    output = np.argmax(model.predict(x, batch_size=BATCH_SIZE), axis=1)
    for p in output:
        pred.append(p)

# display confusion matrix
confusion(y_test, pred)