import random

import matplotlib.pyplot as plt
import numpy as np
from  decimal import *

l = ["A", "B", "C", "D"]

f = open("k.txt", "w")

for line in l:
    f.write(line + '\n')
f.close()
