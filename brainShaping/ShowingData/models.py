import numpy as np
import pandas as pd
''' Loadinf the csv file with the features'''

import csv
with open('finalFeatures.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)