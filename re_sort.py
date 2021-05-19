import numpy as np
import matplotlib.pyplot as plt
import copy
data = np.loadtxt('random_network_spike.txt')
W = np.loadtxt('random_network_W.txt')
data_array = np.loadtxt('random_network.txt')
plt.figure()
plt.scatter(data[:,0],data[:,1],cmap='viridis',linewidth=0.5,color="k",marker='.',s=9,alpha=0.5)
plt.show()

#确定每个event位置
spike_onetime = []
spike_totall = []
index = np.where(data_array.any(axis=1))[0]
singelindex = np.argwhere(data_array[:,5] ==1)
singeldifference = singelindex[1:]- singelindex[0:-1]
difference = index[1:]- index[0:-1]
difference = np.array(difference)
space = np.argwhere(difference>120)#间隔大于多少判定为两个evevt
final = index[np.vstack((space,[[len(index)-1]]))]
start = index[np.vstack(([[0]],space+1))]
mid = index[np.rint((space + np.vstack(([[0]],np.delete(space,-1,0)+1)))/2).astype('int')]
average_fire_time = np.average(final[5:30]-start[5:30])
average_wating_time = np.average(start[6:31]-final[5:30])#群体发放间隔
total = final[-1]-start[0]
singel_space = np.average(singeldifference[5:31])#单个神经元发放间隔

#确定重排顺序
print(data)
data_1 =np.array(data[:,0])
data_l =  [(i and j) for i, j in zip(data_1>int(1380), data_1<int(1400))]
order = (data[data_l,1])
#补齐
if len(order)<data_array.shape[1]:
    print('****',order)
    a = np.arange(data_array.shape[1])
    complement = np.setdiff1d(a,order)
    print(complement)
    order = np.concatenate((order,complement),axis=0)

#画图
data_2 = copy.deepcopy(data[:,1])
data_3 = copy.deepcopy(data[:,1])
for i in range(len(data_2)):
    #print(data_2[i],np.argwhere(order==data_2[i])[0,0])

    data_2[i] = np.argwhere(order==data_2[i])[0,0]

plt.figure()
#plt.plot(np.arange(len(data[:,0])),data[:,0])
#plt.scatter(data[:,0],data_3,cmap='viridis',linewidth=0.5,marker='.',s=60,alpha=1,color='r')
plt.scatter(data[:,0],data_2,cmap='viridis',linewidth=0.5,marker='.',s=60,alpha=1,color='b')
plt.title('spike scatter plot',fontsize='large', fontweight='bold')
plt.xlabel('time/ms',fontsize='large', fontweight='bold')
plt.ylabel('neuron number',fontsize='large', fontweight='bold')
plt.show()
