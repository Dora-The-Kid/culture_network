import reverberation_network_1 as re
import numpy as np
def run(dely,w):
    import matplotlib.pyplot as plt
    n = 80

    Type = np.ones(shape=n)

    T =10000
    dt = 0.0125

    #w = np.loadtxt('Ach_1.txt')
    w = w
    # print(np.sum(w))
    ratio=6300/np.sum(w)
    w = w*ratio

    network = re.network(neuron_number=n,type=Type,w=w,dt=dt,external_W=None,dely_step=dely)

    step =int( T/dt)
    voltage = []
    Ca = []
    X = []
    Y = []
    Z = []
    S = []
    Ge = []
    In = []
    sayrate = []
    satge = []
    mK = []
    spike_array = []
    W_sum = []



    for i in range(step):
        #print(i)


        #if  50.025/dt >i > 50/dt :
        if  i == 50 / dt :
            print('50***')
            network.background_input = np.zeros(shape=n)
            network.background_input[0] = 100/dt
        # elif i == 200/dt:
        #     network.background_input = np.zeros(shape=n)
        #     network.background_input[2:4] = 1000

            #network.background_input = background_input
        # elif i % 150 == 5:
        #     network.background_input = np.array([0/dt,150/dt,0/dt,0/dt])

        else:
            network.background_input= np.zeros(shape=n)


        network.update()
        W_sum.append(np.sum(network.cortical_matrix))
        #print(network.Y[:,0])
        V = network.V
        X.append(network.X[5,0])
        Y.append(network.Y[5,0])
        Z.append(network.Z[5,0])
        S.append(network.S[5,0])
        mK.append(network.mK)
        Ge.append(np.sum(network.ge[5,:]))
        In.append(np.sum(network.increment[5,:]))

        sayrate.append(network.asynrate[5])
        satge.append(np.sum(network.asynge[5,:]))


        Ca.append(network.CaConcen)
        voltage.append(V)
        spike_array.append(network.spike_train_output)

    #print(np.sum(network.cortical_matrix))
    voltage = np.array(voltage)
    spike = network.spike_record_scatter
    spike = np.array(spike)
    spike_array = np.array(spike_array)
    #np.savetxt('Ach_array.txt', spike_array)
    mK = np.array(mK)
    Ca = np.array(Ca)
    # np.savetxt('reverbrtarion_spike_6000ms_2_stdp.txt',spike)
    #np.savetxt('Ach_spike.txt', spike)
    say = np.array(network.asynrate)
    W_sum = np.array(W_sum)
    np.save('W_dely_1ms_stdp',network.cortical_matrix)
    return spike,spike_array,voltage,W_sum

if __name__ == '__main__':
    import network_gen
    #w = network_gen.random_gen(80)
    #np.save('stdp_test',w)
    w = np.load('stdp_test.npy')
    spike,spike_array,voltage,W_sum = run(80,w)
    np.save('spike_stdp',spike)
    np.save('spike_array_stdp',spike_array)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(voltage.shape[0])*0.0125, voltage, alpha=0.3)
    plt.figure()
    plt.scatter(spike[:,0],spike[:,1],cmap='viridis',linewidth=0.5,color="k",marker='.',s=9,alpha=0.5)

    plt.figure()
    plt.plot(np.arange(voltage.shape[0])*0.0125,W_sum)
    plt.show()

