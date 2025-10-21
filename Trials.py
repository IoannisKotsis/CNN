import numpy as np

x=np.array([[1,2,3],
            [5,6,7],
            [8,9,10]])

y=np.array([[0,-1,1],
            [1,1,0],
            [0,0,-1]])

z=np.dot(x,y)
print(z)