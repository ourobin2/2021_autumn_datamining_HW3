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

    # Return the node with the name, create and return new node if not found
    def find(self, name):
        if(not self.contains(name)):
            new_node = Node(name)
            self.nodes.append(new_node)
            return new_node
        else:
            return next(node for node in self.nodes if node.name == name)

    def add_edge(self, parent, child):
        parent_node = self.find(parent)
        child_node = self.find(child)

        parent_node.link_child(child_node)
        child_node.link_parent(parent_node)

    def display(self):
        for node in self.nodes:
            print(f'{node.name} links to {[child.name for child in node.children]}')

    def sort_nodes(self):
        self.nodes.sort(key=lambda node: int(node.name))

    def display_hub_auth(self):
        for node in self.nodes:
            print(f'{node.name}  Auth: {node.old_auth}  Hub: {node.old_hub}')

    def normalize_auth_hub(self):
        auth_sum = sum(node.auth for node in self.nodes)
        hub_sum = sum(node.hub for node in self.nodes)

        for node in self.nodes:
            node.auth /= auth_sum
            node.hub /= hub_sum

    def normalize_pagerank(self):
        pagerank_sum = sum(node.pagerank for node in self.nodes)

        for node in self.nodes:
            node.pagerank /= pagerank_sum

    def get_auth_hub_list(self):
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
        in_neighbors = self.parents
        pagerank_sum = sum((node.pagerank / len(node.children)) for node in in_neighbors)
        random_jumping = d / n
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

    def replace_sim(self):
        self.old_sim = copy.deepcopy(self.new_sim)

    def calculate_SimRank(self, node1, node2):
        # Return 1 if it's same node
        if(node1.name == node2.name):
            return 1.0

        in_neighbors1 = node1.parents
        in_neighbors2 = node2.parents

        # Return 0 if one of them has no in-neighbor
        if(len(in_neighbors1) == 0 or len(in_neighbors2) == 0):
            return 0.0

        SimRank_sum = 0
        for in1 in in_neighbors1:
            for in2 in in_neighbors2:
                SimRank_sum += self.get_sim_value(in1, in2)

        # Follows the equation
        scale = self.decay_factor / (len(in_neighbors1) * len(in_neighbors2))
        new_SimRank = scale * SimRank_sum

        return new_SimRank

    def update_sim_value(self, node1, node2, value):
        node1_idx = self.get_name_index(node1.name)
        node2_idx = self.get_name_index(node2.name)
        self.new_sim[node1_idx][node2_idx] = value

    def get_sim_matrix(self):
        return np.round(np.asarray(self.new_sim), 3)

def init_graph(fname):
    with open(fname) as f:
        lines = f.readlines()

    graph = Graph()

    for line in lines:
        [parent, child] = line.strip().split(',')
        graph.add_edge(parent, child)

    graph.sort_nodes()
    return graph

def HITS_one_iter(graph):
    node_list = graph.nodes

    for node in node_list:
        node.update_auth()

    for node in node_list:
        node.update_hub()

    graph.normalize_auth_hub()


def HITS(graph, iteration):
    for i in range(iteration):
        HITS_one_iter(graph)
        # graph.display_hub_auth()
        # print()

def PageRank_one_iter(graph, d):
    node_list = graph.nodes
    for node in node_list:
        node.update_pagerank(d, len(graph.nodes))
    graph.normalize_pagerank()
    # print(graph.get_pagerank_list())
    # print()


def PageRank(graph, d, iteration):
    for i in range(iteration):
        PageRank_one_iter(graph, d)

def SimRank_one_iter(graph, sim):
    for node1 in graph.nodes:
        for node2 in graph.nodes:
            new_SimRank = sim.calculate_SimRank(node1, node2)
            sim.update_sim_value(node1, node2, new_SimRank)
            # print(node1.label, node2.label, new_SimRank)

    sim.replace_sim()


def SimRank(graph, sim, iteration):
    for i in range(iteration):
        SimRank_one_iter(graph, sim)
        # ans = sim.get_sim_matrix()
        # print(ans)
        # print()


fname = 'graph_1.txt'
path = 'hw3dataset/'+fname
iteration = 100
damping_factor = 0.15
decay_factor = 0.9


graph = init_graph(path)
HITS(graph, iteration)
auth_list, hub_list = graph.get_auth_hub_list()
print(auth_list)
np.savetxt(fname[:-4]+'_authority.txt', auth_list, delimiter=' ', fmt='%.1f')
print(hub_list)
np.savetxt(fname[:-4]+'_hub.txt', hub_list, delimiter=' ', fmt='%.1f')

PageRank(graph, damping_factor, iteration)
print(graph.get_pagerank_list())
np.savetxt(fname[:-4]+'_PageRank.txt', graph.get_pagerank_list(), delimiter=' ', fmt='%.3f')


sim = Similarity(graph, decay_factor)
SimRank(graph, sim, iteration)
ans = sim.get_sim_matrix()
print(ans)
np.savetxt(fname[:-4]+'_SimRank.txt', ans, delimiter=' ', fmt='%.2f')