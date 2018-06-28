import math
from Regression.MultipleLinear import MultipleLinear

data = [[48,68,63],[62,81,72],[79,80,78],[76,83,79],[59,64,62]]
ml = MultipleLinear(data)
print(math.floor(ml.predict([48, 68])) == 61)