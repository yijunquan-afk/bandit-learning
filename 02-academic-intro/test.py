import numpy as np

poker = []
for i in range(10):
    poker.append(i+1)

v = 0
e = 5.5
for x in poker:
    v += (e-x)**2/10
print(v)
