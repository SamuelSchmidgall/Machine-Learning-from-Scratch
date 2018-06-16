import math
from FlexibleNetworks.FlexANN import FlexANN

tups = [([0, 0], [0]),([1, 0], [0]),([0, 1], [0]),([1, 1], [1])]
fn = FlexANN('l_int', tups, [([0, 0], [0]), ([1, 0], [0]),([0, 1], [0]),([1, 1], [1])],
                            [([0, 0], [0]), ([1, 0], [0]),([0, 1], [0]),([1, 1], [1])])

# generic and gate
print(math.floor(fn.forward_propagate([0,0])) == 0.0)
print(math.floor(fn.forward_propagate([0,1])) == 0.0)
print(math.floor(fn.forward_propagate([1,0])) == 0.0)
print(math.ceil(fn.forward_propagate([1,1])) == 1.0)



