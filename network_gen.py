import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import copy





def small_world_network(n,k,p):

    W = np.zeros(shape=(n,n))
    G = nx.DiGraph()
    #G = nx.Graph()
    #G = nx.watts_strogatz_graph(n,k,p)



    for i in range(n):
        from_node = i
        for j in range(1,k+1):
            to_node = i + j
            if to_node >= n:
                to_node =to_node-n
            G.add_edge(from_node, to_node)
    print(G.edges)
    print(G.nodes)

    edges = copy.deepcopy(G.edges())
    for edge in edges:
        if np.random.random() < p:
            G.remove_edge(edge[0], edge[1])
            ran_edge = np.random.randint(n, size=2).tolist()
            G.add_edge(ran_edge[1], ran_edge[0])

    # for node_1 in G.nodes:
    #     for node_2 in G.nodes:
    #         p = np.random.random(1)
    #         if p<0.1 :
    #             G.add_edge(node_1,node_2)




    for u, v in G.edges:
        G.add_edge(u, v, weight=np.random.normal(2.5, 1))
        #G.add_edge(u, v, weight=1)


    ps = nx.circular_layout(G)



        

    for (x,y,w) in G.edges(data=True):
        print(x,y,w)
        W[x-1,y-1] = w['weight']
        #print(w['weight'])
    print(W)
    # W[0:15, 0:15] = W[0:15, 0:15] * 3.5
    # W[0:15, 16:80] = W[0:15, 16:80] * 1.5
    # W[16:80, 0:15] = W[16:80, 0:15] * 0.8
    # W[16:80, 16:80] = W[16:80, 16:80] * 0.8
    plt.figure()
    nx.draw(G, ps, with_labels=False, node_size=30)
    plt.savefig("W_heat_map.png")
    plt.figure()
    W_n = (W-np.average(W))/np.std(W)
    ax = sns.heatmap(W_n, annot=False, center=1, cmap='YlGnBu', vmin=0, vmax=4)
    ax.set_ylim(81, 0)
    ax.set_xlim(0,81)
    plt.savefig("network_fig.png")
    #plt.show()
    return W

def random_gen(n):

    w = np.zeros((n, n), dtype=np.float32)

    p_reccur = np.random.uniform(0, 1, (n, n))
    row, col = np.diag_indices_from(p_reccur)
    p_reccur[row, col] = np.zeros(shape=n)
    w[p_reccur > 0.8] = np.random.normal(3.14, 1.7, size=w[p_reccur > 0.8].shape)


    w = np.abs(w)
    w[0:15, 0:15] = w[0:15, 0:15] * 3.5
    w[0:15, 16:80] = w[0:15, 16:80] * 1.5
    w[16:80, 0:15] = w[16:80, 0:15] * 0.8
    w[16:80, 16:80] = w[16:80, 16:80] * 0.8
    return w

if __name__ == '__main__':
    small_world_network(10,2,0)