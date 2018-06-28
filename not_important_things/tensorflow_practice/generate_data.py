
# Lab 4 Multi-variable linear regression
# import numpy as np

# x = np.random.rand(1000) * 10 - 5
# y = 2 * x**2 + 5 

# data = np.c_[x, y]
# print(data)

# np.savetxt('function_data.csv', data)


# import numpy as np

# data = np.arange(100) # 저장하는 데이터
# np.save('my_data.npy', data) # numpy.ndarray 저장. @파일명, @값
# data2 = np.load('my_data.npy') # 데이터 로드. @파일명


import numpy as np

x = np.random.rand(1000) * 10 - 5
y = 2 * x**2 + 5 

data = np.c_[x, y]
print(data)

np.savetxt('function_data.csv', data)