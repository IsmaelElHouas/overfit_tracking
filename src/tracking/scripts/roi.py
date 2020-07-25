#!/usr/bin/env python

import numpy as np

distance = np.array([[723], [34], [341]])
distance = distance.flatten()
position = np.where(distance<400)
print(position[0])

bboxes = np.array([[842,522,900,634], [568,552,638,756], [371,488,515,784]])
flitered = bboxes[[0, 2],:]
print(flitered)