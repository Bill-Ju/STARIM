import copy
import time
import random
import math
import statistics
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import Pool

random.seed(123)
np.random.seed(123)


class Graph:
    ''' graph class '''
    def __init__(self, nodes, edges, children, parents): 
        self.nodes = nodes # set()
        self.edges = edges # dict{(src,dst): weight, }
        self.children = children # dict{node: set(), }
        self.parents = parents # dict{node: set(), }
        # transfer children and parents to dict{node: list, }
        for node in self.children:
            self.children[node] = sorted(self.children[node])
        for node in self.parents:
            self.parents[node] = sorted(self.parents[node])

        self.num_nodes = len(nodes)
        self.num_edges = len(edges)

        self._adj = None
        self._from_to_edges = None
        self._from_to_edges_weight = None

    def get_children(self, node):
        ''' outgoing nodes '''
        return self.children.get(node, [])

    def get_parents(self, node):
        ''' incoming nodes '''
        return self.parents.get(node, [])

    def get_prob(self, edge):
        return self.edges[edge]

    def get_adj(self):
        ''' return scipy sparse matrix '''
        if self._adj is None:
            self._adj = np.zeros((self.num_nodes, self.num_nodes))
            for edge in self.edges:
                self._adj[edge[0], edge[1]] = self.edges[edge] # may contain weight
            self._adj = csr_matrix(self._adj)
        return self._adj

    def from_to_edges(self):
        ''' return a list of edge of (src,dst) '''
        if self._from_to_edges is None:
            self._from_to_edges_weight = list(self.edges.items())
            self._from_to_edges = [p[0] for p in self._from_to_edges_weight]
        return self._from_to_edges

    def from_to_edges_weight(self):
        ''' return a list of edge of (src, dst) with edge weight '''
        if self._from_to_edges_weight is None:
            self.from_to_edges()
        return self._from_to_edges_weight


def read_graph(path, ind=0, directed=False):
    ''' method to load edge as node pair graph '''
    parents = {}
    children = {}
    edges = {}
    nodes = set()

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not len(line) or line.startswith('#') or line.startswith('%'):
                continue
            row = line.split()
            src = int(row[0]) - ind
            dst = int(row[1]) - ind
            nodes.add(src)
            nodes.add(dst)
            children.setdefault(src, set()).add(dst)
            parents.setdefault(dst, set()).add(src)
            edges[(src, dst)] = 0.0
            if not(directed):
                # regard as undirectional
                children.setdefault(dst, set()).add(src)
                parents.setdefault(src, set()).add(dst)
                edges[(dst, src)] = 0.0

    # change the probability to 1/indegree
    for src, dst in edges:
        edges[(src, dst)] = 1.0 / len(parents[dst])
            
    return Graph(nodes, edges, children, parents),len(nodes)

def computeMC_IC(graph, S, R):
    ''' compute expected influence using MC under IC
        R: number of trials
    '''
    sources = set(S)
    inf = 0
    N = graph.num_nodes
    ret = np.zeros(N)
    for _ in range(R):
        source_set = sources.copy()
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(child for child in graph.get_children(curr_node) \
                    if not(child in source_set) and random.random() <= graph.edges[(curr_node, child)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        inf += len(source_set)
        x = np.zeros(N)
        x[list(source_set)] = 1
        ret += x
    return ret / R

def workerMC(x):
    ''' for multiprocessing '''
    return computeMC_IC(x[0], x[1], x[2]).sum()

def MC_IC_Traj(graph, S, R, steps):
    sources = set(S)
    inf = 0
    N = graph.num_nodes
    traj = np.zeros((steps+1,N))
    for _ in range(R):
        source_set = sources.copy()
        x = np.zeros(N)
        x[list(source_set)] = 1
        queue = deque(source_set)
        traj[0] = x
        for i in range(steps):
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(child for child in graph.get_children(curr_node) \
                    if not(child in source_set) and random.random() <= graph.edges[(curr_node, child)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
            x = np.zeros(N)
            x[list(source_set)] = 1
            traj[i+1] = x
    traj /= R
    return traj


def Worker_MC_IC_Traj(x):
    ''' for multiprocessing '''
    return MC_IC_Traj(x[0], x[1], x[2], x[3])

def computeMC_SIR(graph, S, R, beta, gamma):
    ''' compute expected influence using MC under SIR
        R: number of trials
    '''
    N = graph.num_nodes
    inf = np.zeros(N, dtype=int)
    for _ in range(R):
        states = np.zeros(N, dtype=int)
        states[S] = 1
        max_step = 0
        while max_step<30:
            max_step += 1
            new_states = copy.deepcopy(states)
            count = 0
            for i in range(N):
                if states[i] == 1: 
                    for child in graph.get_children(i):
                        if states[child] == 0 and random.random() <= beta:
                            new_states[child] = 1
                            count += 1
                    if random.random() <= gamma:
                        new_states[i] = 2
                        count += 1
            states = new_states
        inf += (states >= 1)
    return inf / R

def computeMC_SIR_T(graph, S, T, R, beta, gamma):
    ''' compute expected influence using MC under SIR
        R: number of trials
    '''
    N = graph.num_nodes
    inf = np.zeros(N, dtype=int)
    for _ in range(R):
        states = np.zeros(N, dtype=int)
        states[S] = 1
        max_step = 0
        while max_step<30:
            max_step += 1
            new_states = copy.deepcopy(states)
            count = 0
            for i in range(N):
                if states[i] == 1: 
                    for child in graph.get_children(i):
                        if states[child] == 0 and random.random() <= beta:
                            new_states[child] = 1
                            count += 1
                    if random.random() <= gamma:
                        new_states[i] = 2
                        count += 1
            states = new_states
        inf += (states >= 1)
    inf = inf / R
    return inf[T].sum()/N

# def computeMC_SIR(graph, S, R, beta, gamma):
#     ''' 
#     Compute expected influence using Monte Carlo under SIR
#     graph: The graph object (should have a method get_children(node))
#     S: Initial set of infection sources (can be an int or list of nodes)
#     R: Number of Monte Carlo trials
#     beta: Infection rate
#     gamma: Recovery rate
#     '''
#     N = graph.num_nodes
#     inf = np.zeros(N, dtype=int)

#     # Handle single-node source as a list
#     if isinstance(S, int):
#         S = [S]

#     for _ in range(R):
#         # Initialize states: 0 = Susceptible, 1 = Infected, 2 = Recovered
#         states = np.zeros(N, dtype=int)
#         states[S] = 1  # Set initial infection sources
#         active_nodes = set(S)  # Track currently infected nodes

#         # Simulate the process
#         while active_nodes:
#             new_active_nodes = set()
#             for node in active_nodes:
#                 # Attempt to infect neighbors
#                 for neighbor in graph.get_children(node):
#                     if states[neighbor] == 0 and random.random() < beta:
#                         states[neighbor] = 1
#                         new_active_nodes.add(neighbor)

#                 # Attempt to recover the current node
#                 if random.random() < gamma:
#                     states[node] = 2
#                 else:
#                     new_active_nodes.add(node)  # Keep node active if not recovered

#             active_nodes = new_active_nodes  # Update active nodes

#         # Count nodes that were ever infected or recovered
#         inf += (states >= 1)

#     return inf / R  # Return the average influence


def workerMC_SIR(x):
    ''' for multiprocessing '''
    return computeMC_SIR(x[0], x[1], x[2],x[3],x[4])

def workerMC_SIR_T(x):
    ''' for multiprocessing '''
    return computeMC_SIR_T(x[0], x[1], x[2],x[3],x[4],x[5])

def MC_SIR_Traj(graph, S, R, beta, gamma, steps):
    ''' compute expected influence using MC under SIR
        R: number of trials
    '''
    N = graph.num_nodes
    traj = np.zeros((steps+1,N))
    for _ in range(R):
        states = np.zeros(N)
        states[S] = 1
        count_step = 0
        traj[0] += states.copy()
        while count_step<steps:
            count_step += 1
            new_states = states.copy()
            for i in range(N):
                if states[i] == 1: 
                    for child in graph.get_children(i):
                        if states[child] == 0 and random.random() < beta:
                            new_states[child] = 1
                    if random.random() < gamma:
                        new_states[i] = 2
            states = new_states
            traj[count_step] += (states >= 1)
    traj /= R
    return traj

def Worker_MC_SIR_Traj(x):
    ''' for multiprocessing '''
    return MC_SIR_Traj(x[0], x[1], x[2],x[3],x[4], x[5])

# def computeRR_IC(graph, S, T, R, cache=None):
#     ''' compute expected influence using RR under IC
#         R: number of trials
#         The generated RR sets are not saved; 
#         We can save those RR sets, then we can use those RR sets
#             for any seed set
#         cache: maybe already generated list of RR sets for the graph
#         l_c: a list of RR set covered, to compute the incremental score
#             for environment step
#     '''
#     # generate RR set
#     covered = 0
#     generate_RR = False
#     if cache is not None:
#         if len(cache) > 0:
#             # might use break for efficiency for large seed set size or number of RR sets
#             return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
#         else:
#             generate_RR = True

#     for i in range(R):
#         # generate one set
#         source_set = {T[random.randint(0, len(T))]}
#         queue = deque(source_set)
#         while True:
#             curr_source_set = set()
#             while len(queue) != 0:
#                 curr_node = queue.popleft()
#                 curr_source_set.update(parent for parent in graph.get_parents(curr_node) \
#                     if not(parent in source_set) and random.random() <= graph.edges[(parent, curr_node)])
#             if len(curr_source_set) == 0:
#                 break
#             queue.extend(curr_source_set)
#             source_set |= curr_source_set
#         # compute covered(RR) / number(RR)
#         for s in S:
#             if s in source_set:
#                 covered += 1
#                 break
#         if generate_RR:
#             cache.append(source_set)
#     return covered * 1.0 / R * graph.num_nodes

# def computeRR_SIR(graph, T, R, beta, gamma, cache=None):
#     ''' compute expected influence using RR under IC
#         R: number of trials
#         The generated RR sets are not saved; 
#         We can save those RR sets, then we can use those RR sets
#             for any seed set
#         cache: maybe already generated list of RR sets for the graph
#         l_c: a list of RR set covered, to compute the incremental score
#             for environment step
#     '''
#     # generate RR set
#     covered = 0
#     generate_RR = False
#     if cache is not None:
#         if len(cache) > 0:
#             # might use break for efficiency for large seed set size or number of RR sets
#             return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
#         else:
#             generate_RR = True
#     N = graph.num_nodes
#     for i in range(R):
#         # generate one set
#         S = T[random.randint(0, len(T))]
#         states = np.zeros(N, dtype=int)
#         states[S] = 1
#         max_step = 0
#         while max_step<30:
#             max_step += 1
#             new_states = copy.deepcopy(states)
#             count = 0
#             for i in range(N):
#                 if states[i] == 1: 
#                     for child in graph.get_children(i):
#                         if states[child] == 0 and random.random() <= beta:
#                             new_states[child] = 1
#                             count += 1
#                     if random.random() <= gamma:
#                         new_states[i] = 2
#                         count += 1
#             states = new_states
#         inf += (states >= 1)
        
        
#         for s in S:
#             if s in source_set:
#                 covered += 1
#                 break
#         if generate_RR:
#             cache.append(source_set)
#     return covered * 1.0 / R * graph.num_nodes


# def workerRR(x):
#     ''' for multiprocessing '''
#     return computeRR(x[0], x[1], x[2])

