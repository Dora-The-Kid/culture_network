import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
#
# # path = os.path.join('D:\\Renyi\\culture_network\\xiaoweoshijie\\slurm_data\\Ach','_epsc0.2_gl2.0.csv')
# # df = pd.read_csv(path,header=0)
# # df = df.drop('trail', axis=1)
# # print(df)
# # print(isinstance(df.iloc[0].iat[1], str))
# # mask = df.apply(lambda r : any([isinstance(e, str) for e in r  ]),axis=1)
# # print(mask)
#
# A = [0,1]
# B = [2,0]
# C = [-2,0]
# x = [A[0], B[0], C[0], A[0]]
# y = [A[1], B[1], C[1], A[1]]
# plt.figure()
# ax = plt.gca()
# ax.plot(x, y, linewidth=2)
# a = np.array([3,4])
# plt.plot(a[0],a[1],'*')
#
#
# revert = np.array([[0,-1],[1,0]])
# A = np.dot(A,revert)
# B = np.dot(B,revert)
# C = np.dot(C,revert)
# print(A,B,C)
# x = [A[0], B[0], C[0], A[0]]
# y = [A[1], B[1], C[1], A[1]]
# a = np.dot(a,revert)
# plt.figure()
# ax = plt.gca()
# ax.plot(x, y, linewidth=2)
# plt.plot(a[0],a[1],'*')
# plt.show()

#
# x = torch.arange(64).view(8,8)
# print(x)
# a = x.unfold(0,3,1).unfold(1,3,1)
# print(a.shape)
# print(a)
import seaborn as sns
x = np.linspace(-10, 10, 100)
y = x**2
plt.figure()
plt.plot(x, y)

# 设置轴的刻度
#plt.xticks(range(-8, 8, 2))
#plt.yticks([0, 40, 60], ["bad", 'good', "best"])

plt.grid(b= True,color='r', linestyle='--', linewidth=0.5,alpha=0.3)

plt.show()