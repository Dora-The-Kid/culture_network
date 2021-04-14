import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad
import matplotlib.pyplot as plt
import numpy

def simulator(gl):
    V1 = -1.2
    V2 = 18
    V3 = 2
    V4 = 30
    Iap = 100
    gca = 4.4
    Vca = 120
    gk = 8
    VK = -84
    gl = gl#2
    Vl =-60
    dt = 0.01
    Vot_Threshold = 0
    init_V, init_W, time, substep, = -26, 0.1135, 300, 0.2


    rV = init_V
    rW = init_W
    rV_1 = 0
    rW_1 = 0
    i = 0

    for step in range(1,int(time/substep)):
        dv = (1 / 20) * (Iap - gca * ((1+np.tanh((rV-V1)/V2))/2) * (rV - Vca) - gl * (rV - Vl) - rW * (gk * (rV - VK)))
        dw = 0.04 * ((0.5 * (1 + np.tanh((rV - V3) / (V4))) - rW) / (1/np.cosh((rV-V3)/(2*V4))))
        rV_1 = rV
        rW_1 = rW
        rV = rV+dv
        rW = rW+dw


        if rV_1<Vot_Threshold and rV>Vot_Threshold:
            delt_t = 2 * substep*(0. - rV_1) / (rV - rV_1)
            i +=1
            if i==2:
                start = step*substep+delt_t
            if i ==7:
                final = step*substep+delt_t
    return final-start


def simulator_1(gl,color):
    V1 = -1.2
    V2 = 18
    V3 = 2
    V4 = 30
    Iap = 100
    gca = 4.4
    Vca = 120
    gk = 8
    VK = -84
    gl = gl#2
    Vl =-60
    dt = 0.01
    Vot_Threshold = 0
    init_V, init_W, time, substep, = -26, 0.1135, 300, 0.2


    rV = init_V
    rW = init_W
    rV_1 = 0
    rW_1 = 0
    i = 0
    total = []


    for step in range(1,int(time/substep)):
        dv = (1 / 20) * (Iap - gca * ((1+np.tanh((rV-V1)/V2))/2) * (rV - Vca) - gl * (rV - Vl) - rW * (gk * (rV - VK)))
        dw = 0.04 * ((0.5 * (1 + np.tanh((rV - V3) / (V4))) - rW) / (1/np.cosh((rV-V3)/(2*V4))))
        rV_1 = rV
        rW_1 = rW
        rV = rV+dv
        rW = rW+dw
        total.append(rV)


        if rV_1<Vot_Threshold and rV>Vot_Threshold:
            delt_t = 2 * substep*(0. - rV_1) / (rV - rV_1)
            i +=1
            if i==2:
                start = step*substep+delt_t
            if i ==7:
                final = step*substep+delt_t
    plt.plot(numpy.arange(int(len(total))),total,c = color,alpha = 0.7,label = str(gl))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)


    return final-start



grad_si = grad(simulator)
print(simulator(2.))
print(grad_si(2.))
plt.figure()
simulator_1(2.,'r')
simulator_1(1.95,'b')
simulator_1(1.9,'y')
plt.xlabel('time/step')
plt.ylabel('voltage/mv')

plt.show()

