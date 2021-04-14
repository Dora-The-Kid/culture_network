import numpy as np
import matplotlib.pyplot as plt

V1 =-1.2
V2 =19
V3 =2
V4 =30
Iap =100
gca =4.4
Vca =120
gk =8
VK =-84
gl = 2
Vl = -60
dt = 0.01

def myarctah(x):
    return 0.5*np.log((1+x)/(1-x))
def fun_tau(V):
    return 1/np.cosh((V-V3)/(2*V4))

def fun_m(V):
    return  (1+np.tanh((V-V1)/V2))/2

def get_V_nullcline(V):
  """
  Solve for W
  """
  return(Iap - gca*fun_m(V)*(V-Vca)-gl*(V-Vl))/(gk*(V-VK))

def get_W_nullcline(W):
    """
    Solve for U
    """

    return(myarctah(2*(W)-1))*2*V4+V1


def plot_nullclines(V_null_V, V_null_W, W_null_V, W_null_W):

  plt.plot(V_null_V, V_null_W, 'b', label='V nullcline')
  plt.plot(W_null_V, W_null_W, 'r', label='W nullcline')
  plt.xlabel(r'V')
  plt.ylabel(r'W')
  plt.legend(loc='best')
  #plt.show()

def plot_vector(rV,rW,drV,drW,n_skip):
    plt.quiver(rV[::n_skip, ::n_skip], rW[::n_skip, ::n_skip],
               drV[::n_skip, ::n_skip], drW[::n_skip, ::n_skip],
               angles='xy', scale_units='xy', scale=10., facecolor='c')

def vector(V,W):
    dV =Iap - gca*fun_m(V)*(V-Vca)-gl*(V-Vl)-W*(gk*(V-VK))
    dw = 0.5*(1+np.tanh((V-V3)/(V4)))/fun_tau(V)
    return dV,dw

def simulator(init_V,init_W,time,substep,color):
    rV =[]
    rW = []
    V = init_V
    W = init_W
    for step in range(int(time/substep)):
        dz = MorrisLecar_function(V,W)
        dV = dz[0]
        dW = dz[1]
        V = V+dV
        W = W+dW
        rV.append(V)
        rW.append(W)
    rV = np.array(rV)
    print(np.sum((rV[1:]>0)*(rV[:-1]<0)))
    plt.plot(rV,rW,alpha=0.5,c = color)
    return rV



def MorrisLecar_function(V,W):
    dv = (1/20)*(Iap - gca*fun_m(V)*(V-Vca)-gl*(V-Vl)-W*(gk*(V-VK)))
    dw = 0.04*((0.5*(1+np.tanh((V-V3)/(V4)))-W)/fun_tau(V))
    return np.array([dv,dw])
def nullclines(vlims,wlims,stp):
    v = np.arange(vlims[0], vlims[1], stp)
    w = np.arange(wlims[0], wlims[1], stp)
    V,W =  np.meshgrid(v,w)
    plt.figure()
    plt.contour(V,W,MorrisLecar_function(V,W)[0,::],0,c = 'b')
    plt.contour(V, W, MorrisLecar_function(V, W)[1, ::],0, c = 'g')

def phasearrows(vlims,wlims,stp):
    v = np.arange(vlims[0], vlims[1], stp)
    w = np.arange(wlims[0], wlims[1], stp)
    V, W = np.meshgrid(v, w)
    z1,z2 = MorrisLecar_function(V, W)
    max_z1 = np.max(z1)
    max_z2 = np.max(z2)
    dt = np.min(np.array([np.abs(vlims[1]-vlims[0]) / max_z1, np.abs(wlims[1]-wlims[0]) / max_z2]))
    lens = np.sqrt(np.square(z1)+np.square(z2))
    lens2 = lens/np.max(lens)
    dv = dt*z1/(lens2+0.1)
    dw = dt*z2/(lens2+0.1)
    print(V[::,::50].shape)
    plt.quiver(V[::3,::200],W[::3,::200],dv[::3,::200],dw[::3,::200],angles="xy",facecolor='black',scale=50,scale_units='xy',alpha = 0.3)



if __name__ == '__main__':
    a = nullclines([-70,50],[0,1],0.01)
    phasearrows([-70,50],[0,1],0.01)
    rv1 = simulator(-26,0.1140,300,0.2,'r')
    rv2 = simulator(-26, 0.1134, 300, 0.2, 'b')
    plt.figure()
    plt.xlabel('time/ms')
    plt.ylabel('voltage/mv')
    plt.plot(np.arange(int(300 / 0.2)) * 0.2, rv1, color='r')
    plt.plot(np.arange(int(300 / 0.2)) * 0.2, rv2, color='b')



    V_null_V = np.linspace(-60, 40, 1000)
    W_null_W = np.linspace(0, 1, 1000)
    V_null_W = get_V_nullcline(V_null_V)
    W_null_V = get_W_nullcline(W_null_W)
    rV, rW = np.meshgrid(V_null_V, W_null_W)
    drV, drW = vector(rV, rW)
    n_skip = 150
    # plt.figure()
    # plt.xlim((-70, 50))
    # plt.ylim((0,0.6))
    #
    # plot_nullclines(V_null_V,V_null_W,W_null_V,V_null_W)
    # plot_vector(rV, rW, drV, drW,n_skip)
    plt.show()

