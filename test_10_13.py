import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
import pandas as pd
data_0 = np.array(
    [
318046,
315573,
321017,
304790,
372767,

311894,
303780




    ]
)*0.0125
data_1 = np.array([307263,
300547,
371554,
342318,
335241,
328635,
324150,
338215




])*0.0125
data_2 = np.array([322581,
323837,
306339,
305075,
303732,
322913,
322309




])*0.0125
data_3 = np.array([
307466,
295292,
305653,
281254,
280097,
319970,
267692



])*0.0125
data = [
    data_0,data_1,data_2,data_3
]

a =stats.levene(data_1, data_2)
b = stats.ttest_ind(data_1,data_2,equal_var=True)
plt.figure()
plt.boxplot(x = data,
            labels = ['0.19','0.2','0.22','0.25'], # 添加具体的标签名称
            showmeans=True,
            patch_artist=True,
            boxprops = {'color':'black','facecolor':'#9999ff'},
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},
            meanprops = {'marker':'D','markerfacecolor':'indianred'},
            medianprops = {'linestyle':'--','color':'orange'})

plt.title('duration')
plt.show()
print(a,b)
def plot_sig(xstart,xend,ystart,yend,sig):
    for i in range(len(xstart)):
        x = np.ones((2))*xstart[i]
        y = np.arange(ystart[i],yend[i],yend[i]-ystart[i]-0.1)
        plt.plot(x,y,label="$y$",color="black",linewidth=1)

        x = np.arange(xstart[i],xend[i]+0.1,xend[i]-xstart[i])
        y = yend[i]+0*x
        plt.plot(x,y,label="$y$",color="black",linewidth=1)

        x0 = (xstart[i]+xend[i])/2
        y0=yend[i]
        plt.annotate(r'%s'%sig, xy=(x0, y0), xycoords='data', xytext=(-15, +1),
                     textcoords='offset points', fontsize=16,color="red")
        x = np.ones((2))*xend[i]
        y = np.arange(ystart[i],yend[i],yend[i]-ystart[i]-0.1)
        plt.plot(x,y,label="$y$",color="black",linewidth=1)
        plt.ylim(0,math.ceil(max(yend)+4))             #使用plt.ylim设置y坐标轴范围
    #     plt.xlim(math.floor(xstart)-1,math.ceil(xend)+1)
        #plt.xlabel("随便画画")         #用plt.xlabel设置x坐标轴名称
        '''设置图例位置'''
        #plt.grid(True)
    plt.show()
plot_sig([0.42,1.42],[1.42,2.42],[30,20],[30.8,20.8],'***')