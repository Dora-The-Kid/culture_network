import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
PATH = ''
def load_csv(path):
    df = pd.read_csv(path,header=0,dtype=str)

    df.drop('trail', axis=1, inplace=True)
    print(df)

    df = df[~ df['total'].str.contains('None')]
    print(path)


    #df = df[ ~ df['spike_times'].str.contains('None')]
    df = df.apply(pd.to_numeric, errors='ignore')
    reverbration_rate = len(df)/20
    spike_times = df['spike_times'].to_numpy()
    reverbration_duratuion = df['total'].to_numpy()


    return reverbration_rate,np.average(reverbration_duratuion),np.average(spike_times)
def drow(data):
    _x = np.arange(6)*0.01+0.2
    _y = np.arange(5)*0.025+1.9
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    bottom = np.zeros_like(data)
    width = 0.005
    depth = 0.01
    data = np.array(data).reshape((6, 5))
    data[4,4] = 0.8


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0, 6):
        for h in range(0, 5):
            color = np.array([255, 255, 10*i+h])/255

            ax.bar3d(i*0.01+0.2, h*0.025+1.9, bottom, width, depth, data[i,h], alpha = 0.1)

    plt.title('reverberation occurrence probability')
    ax.set_ylabel('gl')
    ax.set_xlabel('epsc')

    data =(data-np.mean(data))/np.std(data)
    print(data)
    import seaborn as sns
    plt.figure()
    sns.set()
    yticklabels = yticks = np.linspace(0, 10, 1) /10


    ax = sns.heatmap(data, annot=False, center=0, cmap='YlGnBu', vmin=-1, vmax=1)
    ax.set_ylim(7, 0)
    ax.set_xlim(0, 7)

    ax.set_ylabel('epsc')
    ax.set_xlabel('gl')


    plt.title('reverberation_occurrence', fontsize='large', fontweight='bold')


def main():
    reverbration_rate_array, reverbration_duratuion_array, spike_times_array = [],[],[]
    for i in range(0, 6):
        for h in range(0, 5):
            epsc = 0.2 + i * 0.01
            gl = 1.9 + h * 0.025
            name = '_epsc' + str(round(epsc, 3)) + '_gl' + str(round(gl, 3))+'.csv'
            name =os.path.join('D:\\Renyi\\culture_network\\xiaoweoshijie\\slurm_data\\Ach',name)
            reverbration_rate,reverbration_duratuion,spike_times = load_csv(name)
            reverbration_duratuion_array.append(reverbration_duratuion)
            reverbration_rate_array.append(reverbration_rate)
            spike_times_array.append(spike_times)
    reverbration_duratuion_array = np.array(reverbration_duratuion_array)
    print(reverbration_duratuion_array.shape)
    #reverbration_duratuion_array = np.average(reverbration_duratuion_array,1)
    drow(reverbration_rate_array)

    plt.show()

if __name__ == '__main__':
    main()




