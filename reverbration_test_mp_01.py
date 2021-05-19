import reverberation_network_1 as re
import numpy as np
def run(dely,w):
    import matplotlib.pyplot as plt
    n = 80

    Type = np.ones(shape=n)

    T =7000
    dt = 0.0125

    #w = np.loadtxt('Ach_1.txt')
    w = w
    # print(np.sum(w))
    ratio=6100/np.sum(w)
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
    return spike,spike_array,voltage
