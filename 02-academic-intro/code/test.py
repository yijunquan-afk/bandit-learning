import numpy as np

def random_argmax(vector):
    """随机选择argmax"""
    index = np.random.choice(np.where(vector == vector.max())[0])
    return index


vect = np.array([1,301,21,45,301])
print(np.where(vect == vect.max())[0])
print(np.random.choice(vect))
print(np.random.rand(3))