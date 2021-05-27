import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray
def Kendall_tau(a,b):
    #a = [1, 1, 2, 2, 5, 5, 8, 8, 9, 10]
    #b = [2, 7, 2, 3, 3, 6, 8, 4, 5, 5]
    # a = [12,2,1,12,2]
    # b = [1,4,7,1,0]
    Lens = len(a)

    ties_onlyin_x = 0
    ties_onlyin_y = 0
    con_pair = 0
    dis_pair = 0
    for i in range(Lens - 1):
        for j in range(i + 1, Lens):
            test_tying_x = np.sign(a[i] - a[j])
            test_tying_y = np.sign(b[i] - b[j])
            panduan = test_tying_x * test_tying_y
            if panduan == 1:
                con_pair += 1
            elif panduan == -1:
                dis_pair += 1

            if test_tying_y == 0 and test_tying_x != 0:
                ties_onlyin_y += 1
            elif test_tying_x == 0 and test_tying_y != 0:
                ties_onlyin_x += 1

    Kendallta1 = (con_pair - dis_pair) / np.sqrt((con_pair + dis_pair + ties_onlyin_x) * (dis_pair + con_pair + ties_onlyin_y))
    return Kendallta1

def read_file(spike_adress,spike_array_adress):
    spike = np.load(spike_adress)
    spike_array = np.load(spike_array_adress)
    print(spike.shape,spike_array.shape)
    if spike_array[spike_array==1].shape[0] <= 80 :
        print('######')
        return ['None','None','None','None','None']
    # 确定每个event位置
    spike_onetime = []
    spike_totall = []
    index = np.where(spike_array.any(axis=1))[0]

    difference = index[1:] - index[0:-1]
    difference = np.array(difference)
    space = np.argwhere(difference > int(20/0.0125))  # 间隔大于多少判定为两个event
    final = (index[np.vstack((space, [[len(index) - 1]]))]+5)*0.0125
    start = (index[np.vstack(([[0]], space + 1))]-5)*0.0125
    matrix = []
    plt.figure()
    plt.scatter(spike[:, 0], spike[:, 1], cmap='viridis', linewidth=0.5, color="k", marker='.', s=9, alpha=0.5)
    plt.vlines(final , ymax=80, ymin=0, color='r', alpha=0.5)
    plt.vlines(start , ymax=80, ymin=0, color='b', alpha=0.5)
    plt.show()
    for i in range(len(final)):
        print(((final[i]>spike[:,0]) * (spike[:,0]>start[i])))

        matrix.append(spike[((final[i]>spike[:,0]) * (spike[:,0]>start[i])),1])
    fig = np.zeros(shape=(len(final),len(final)))
    print(len(matrix[0]),len(matrix[22]))
    for i in range(len(matrix)):
        for h in range( len(matrix)):
            fig[i,h] = Kendall_tau(matrix[i],matrix[h])
    return fig
def drow(fig:ndarray):
    plt.figure()
    sns.heatmap(fig,annot=False, center=0.25, cmap='YlGnBu', vmin=0, vmax=1)
    plt.ylim(fig.shape[1]+2,0)
    plt.xlim(0,(fig.shape[0])+2)
    plt.title('Kendall_tau')
if __name__ == '__main__':
    fig = read_file(spike_adress='spike_stdp.npy',spike_array_adress='spike_array_stdp.npy')
    drow(fig)
    plt.show()
    




