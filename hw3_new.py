import numpy as np
import copy

class Graph:
    def __init__(self):
        self.nodes = []

    def contains(self, name):
        for node in self.nodes:
            if(node.name == name):
                return True
        return False

    def find(self, name):
        if(not self.contains(name)):
            new_node = Node(name)
            self.nodes.append(new_node)
            return new_node
        else:
            return next(node for node in self.nodes if node.name == name)

    def new_edge(self, parent, child):
        parent_node = self.find(parent)
        child_node = self.find(child)

        parent_node.link_child(child_node)
        child_node.link_parent(parent_node)

    def normalize_ah(self):
        auth_sum = sum(node.auth for node in self.nodes)
        hub_sum = sum(node.hub for node in self.nodes)

        for node in self.nodes:
            node.auth /= auth_sum
            node.hub /= hub_sum

    def normalize_pagerank(self):
        pagerank_sum = sum(node.pagerank for node in self.nodes)

        for node in self.nodes:
            node.pagerank /= pagerank_sum

    def show(self):
        for node in self.nodes:
            print(f'{node.name} links to {[child.name for child in node.children]}')

    def nodes_sort(self):
        self.nodes.sort(key=lambda node: int(node.name))

    def show_hub_auth(self):
        for node in self.nodes:
            print(f'{node.name}  Auth: {node.old_auth}  Hub: {node.old_hub}')


    def get_ah_list(self):
        auth_list = np.asarray([node.auth for node in self.nodes], dtype='float32')
        hub_list = np.asarray([node.hub for node in self.nodes], dtype='float32')

        return np.round(auth_list, 3), np.round(hub_list, 3)

    def get_pagerank_list(self):
        pagerank_list = np.asarray([node.pagerank for node in self.nodes], dtype='float32')
        return np.round(pagerank_list, 3)

class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parents = []
        self.auth = 1.0
        self.hub = 1.0
        self.pagerank = 1.0

    def link_child(self, new_child):
        for child in self.children:
            if(child.name == new_child.name):
                return None
        self.children.append(new_child)

    def link_parent(self, new_parent):
        for parent in self.parents:
            if(parent.name == new_parent.name):
                return None
        self.parents.append(new_parent)

    def update_auth(self):
        self.auth = sum(node.hub for node in self.parents)

    def update_hub(self):
        self.hub = sum(node.auth for node in self.children)

    def update_pagerank(self, d, n):
        near = self.parents
        random_jumping = d / n
        pagerank_sum = sum((node.pagerank / len(node.children)) for node in near)
        self.pagerank = random_jumping + (1-d) * pagerank_sum


    

    
class Similarity:
    def __init__(self, graph, decay_factor):
        self.decay_factor = decay_factor
        self.name_list, self.old_sim = self.init_sim(graph)
        self.node_num = len(self.name_list)
        self.new_sim = [[0] * self.node_num for i in range(self.node_num)]

    def init_sim(self, graph):
        nodes = graph.nodes
        name_list = [node.name for node in nodes]
        sim = []
        for name1 in name_list:
            temp_sim = []
            for name2 in name_list:
                if(name1 == name2):
                    temp_sim.append(1)
                else:
                    temp_sim.append(0)
            sim.append(temp_sim)

        return name_list, sim

    def get_name_index(self, name):
        return (self.name_list.index(name))

    def get_sim_value(self, node1, node2):
        node1_idx = self.get_name_index(node1.name)
        node2_idx = self.get_name_index(node2.name)
        return self.old_sim[node1_idx][node2_idx]

    def update_sim_value(self, node1, node2, value):
        node1_idx = self.get_name_index(node1.name)
        node2_idx = self.get_name_index(node2.name)
        self.new_sim[node1_idx][node2_idx] = value

    def get_sim_matrix(self):
        return np.round(np.asarray(self.new_sim), 3)

    def replace_sim(self):
        self.old_sim = copy.deepcopy(self.new_sim)

    def cal_simrank(self, node1, node2):
        if(node1.name == node2.name):
            return 1.0

        near1 = node1.parents
        near2 = node2.parents

        if(len(near1) == 0 or len(near2) == 0):
            return 0.0

        simrank_sum = 0
        for in1 in near1:
            for in2 in near2:
                simrank_sum += self.get_sim_value(in1, in2)

        scale = self.decay_factor / (len(near1) * len(near2))
        new_simrank = scale * simrank_sum

        return new_simrank

    

def init_graph(fname):
    with open(fname) as f:
        lines = f.readlines()

    graph = Graph()

    for line in lines:
        [parent, child] = line.strip().split(',')
        graph.new_edge(parent, child)

    graph.nodes_sort()
    return graph

def HITS_one_iter(graph):
    node_list = graph.nodes

    for node in node_list:
        node.update_auth()

    for node in node_list:
        node.update_hub()

    graph.normalize_ah()


def HITS(graph, iteration):
    for i in range(iteration):
        HITS_one_iter(graph)


def PageRank_one_iter(graph, d):
    node_list = graph.nodes
    for node in node_list:
        node.update_pagerank(d, len(graph.nodes))
    graph.normalize_pagerank()


def PageRank(graph, d, iteration):
    for i in range(iteration):
        PageRank_one_iter(graph, d)

def simrank_one_iter(graph, sim):
    for node1 in graph.nodes:
        for node2 in graph.nodes:
            new_simrank = sim.cal_simrank(node1, node2)
            sim.update_sim_value(node1, node2, new_simrank)

    sim.replace_sim()


def simrank(graph, sim, iteration):
    for i in range(iteration):
        simrank_one_iter(graph, sim)



fname = 'graph_3.txt'
path = 'hw3dataset/'+fname
iteration = 6
damping_factor = 0.10
decay_factor = 0.9


graph = init_graph(path)
HITS(graph, iteration)
auth_list, hub_list = graph.get_ah_list()
print("iteration:",iteration)
print("damping factor:",damping_factor)
print("decay factor:",decay_factor)

print("\nAuthority:")
print(auth_list)
np.savetxt(fname[:-4]+'_authority.txt', auth_list, delimiter=' ', fmt='%.3f')
print("\nHub:")
print(hub_list)
np.savetxt(fname[:-4]+'_hub.txt', hub_list, delimiter=' ', fmt='%.3f')

PageRank(graph, damping_factor, iteration)
print("\nPageRank:")
print(graph.get_pagerank_list())
np.savetxt(fname[:-4]+'_PageRank.txt', graph.get_pagerank_list(), delimiter=' ', fmt='%.3f')


sim = Similarity(graph, decay_factor)
simrank(graph, sim, iteration)
ans = sim.get_sim_matrix()
print("\nSimeRank:")
print(ans)
np.savetxt(fname[:-4]+'_SimRank.txt', ans, delimiter=' ', fmt='%.3f')