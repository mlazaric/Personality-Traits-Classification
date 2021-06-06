import numpy as np
import csv

from glob import glob
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

mask = np.array(Image.open("mask.png"))
for filename in glob('*.csv'):
    frequencies = {}
    with open(filename) as f:
        for row in csv.DictReader(f):
            frequencies[row['name']] = float(row['score'])

    wc = WordCloud(
        background_color='white',
        mask=mask,
        colormap='tab10'
    )
    wc.generate_from_frequencies(frequencies)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.savefig(filename[:-3] + 'png')
