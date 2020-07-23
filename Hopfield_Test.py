import numpy as np
import Hopfield as Hop
import matplotlib as mtpl

from matplotlib import pyplot as plt


# Matrix version of one
One_Matrix = np.full((10, 10), -1)
One_Matrix[3, 4] = One_Matrix[2:9, 5] = 1
# Matrix Version of disturbed one
disturbed_One_Matrix = np.full((10, 10), -1)
disturbed_One_Matrix[3, 4] = disturbed_One_Matrix[3:9, 5] = 1
'''
plt.matshow(One_Matrix)
plt.show()
plt.matshow(disturbed_One_Matrix)
plt.show()
'''
# Matrix Version of two
Two_Matrix = np.full((10, 10), -1)
Two_Matrix[2, 4:6] = Two_Matrix[3, 3] = Two_Matrix[3:5, 6] = Two_Matrix[8, 3:7] = Two_Matrix[7, 3] = Two_Matrix[6, 4] = Two_Matrix[5, 5] = 1
# Matrix Version of disturbed two
disturbed_Two_Matrix = np.full((10, 10), -1)
disturbed_Two_Matrix[2, 4:6] = disturbed_Two_Matrix[3, 3] = disturbed_Two_Matrix[3:5, 6] = disturbed_Two_Matrix[8, 3:7] = disturbed_Two_Matrix[7, 3] = disturbed_Two_Matrix[6, 4] = 1
'''
plt.matshow(Two_Matrix)
plt.show()
plt.matshow(disturbed_Two_Matrix)
plt.show()
'''
# Matrix Version of three
Three_Matrix = np.full((10, 10), -1)
Three_Matrix[0, 4:6] = Three_Matrix[1, 3] = Three_Matrix[1:3, 6] = Three_Matrix[3, 5] = Three_Matrix[4, 4] = Three_Matrix[5, 5] = Three_Matrix[6:8, 6] = Three_Matrix[8, 4:6] = Three_Matrix[7, 3] = 1
# Matrix Version of disturbed three
disturbed_Three_Matrix = np.full((10, 10), -1)
disturbed_Three_Matrix[0, 4:6] = disturbed_Three_Matrix[1, 3] = disturbed_Three_Matrix[1:3, 6] = disturbed_Three_Matrix[3, 5] = disturbed_Three_Matrix[4, 4] = disturbed_Three_Matrix[6:8, 6] = disturbed_Three_Matrix[8, 4:6] = disturbed_Three_Matrix[7, 3] = 1
'''
plt.matshow(Three_Matrix)
plt.show()
plt.matshow(disturbed_Three_Matrix)
plt.show()
'''
Training_Data = np.concatenate((One_Matrix.reshape(100, 1), Two_Matrix.reshape(100, 1), Three_Matrix.reshape(100, 1)), axis=1)
Network = Hop.HopfieldNet(100)
Network.train(Training_Data)
associated_pattern = Network.recpattern(disturbed_One_Matrix.reshape((100, 1)), 30)
plt.matshow(associated_pattern.reshape(10, 10))
plt.show()