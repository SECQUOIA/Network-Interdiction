import networkx as nx
# import dwave_networkx as dnx
import numpy as np
from gurobipy import *
import os.path
import neal, dimod
import subprocess
import matplotlib.pyplot as plt
import solve_qubo_via_dwave
import random
from pyqubo import Binary

def read_flow_file(filename):
    f = open(filename, 'r')
    g = nx.DiGraph()
    lines = f.readlines()
    f.close()

    num_nodes = int(lines[0])
    second_line = lines[1]
    second_line_words = second_line.split()
    network_src_node = int(second_line_words[0])
    network_tgt_node = int(second_line_words[1])

    g.add_nodes_from(range(num_nodes))
    for line in lines[2:]:
        words = line.split()
        src_node = int(words[0])
        tgt_node = int(words[1])
        weight = int(words[2])
        g.add_edge(src_node, tgt_node, weight=weight)
    
    return g, network_src_node, network_tgt_node

def leda_to_flow_file(num_nodes):
    inputfile = open(f'./testcases/{num_nodes}.u', 'r')
    for i in range(3):
        inputfile.readline()
    if num_nodes != int(inputfile.readline()):
        print("Input Graph does not match desired node count..")
        exit(1)
    for i in range(num_nodes):
        inputfile.readline()
    num_edges = int(inputfile.readline())
    g = nx.Graph()
    g.add_nodes_from(list(range(num_nodes)))

    for i in range(num_edges):
        line = inputfile.readline()
        strings = line.split()
        g.add_edge(int(strings[0])-1, int(strings[1])-1)

    if nx.is_planar(g) == False:
        print("The grpah is not planar?")
        exit(1)

    #keep adding directed edges until the graph is SCC, also since the planar graph is already maximal, we cannot add *new* edges but just add directions to exisiting edges
    di_g = nx.DiGraph()
    for e in g.edges():
        coin_flip = random.randint(0, 1)
        if coin_flip == 0:
            di_g.add_edge(e[0], e[1])
        else:
            di_g.add_edge(e[1], e[0])
    strong_connected = nx.is_strongly_connected(di_g)    

    while not strong_connected:
        for e in di_g.edges():
            if (e[1], e[0]) not in di_g.edges():
                di_g.add_edge(e[1], e[0])
        strong_connected = nx.is_strongly_connected(di_g)
    
    for e in di_g.edges():
        # add random edge weights (capacities) from 15-100
        di_g.edges[e]['weight'] = np.random.randint(15,100)

    sorted_nodes = sorted(di_g.degree, key=lambda x: x[1], reverse=True)
    src_node = sorted_nodes[0][0]
    tgt_node = sorted_nodes[1][0]

    file_to_write = open(f'./testcases/{num_nodes}.planarflow', 'w')
    file_to_write.write(f"{nx.number_of_nodes(di_g)}\n")
    file_to_write.write(f"{src_node} {tgt_node}\n")
    for e in di_g.edges():
        file_to_write.write(f"{e[0]} {e[1]} {di_g[e[0]][e[1]]['weight']}\n")
    file_to_write.close()

def create_random_nn_graphs(n, k, seed=42):
    g = nx.random_geometric_graph(n, 0, 2, seed=seed)

    for i in g.nodes():
        dist_array = []
        for j in g.nodes():
            curr_dist = np.linalg.norm(np.asarray(g.nodes[i]['pos']) - np.asarray(g.nodes[j]['pos']))
            dist_array.append(curr_dist)
        
        idx = np.argpartition(np.asarray(dist_array), k+1)
        for m in range(k):
            g.add_edge(i, idx[m+1])

        # g.add_edge(i,neighbor)
    
    di_g = nx.DiGraph()
    for e in g.edges():
        coin_flip = random.randint(0, 1)
        if coin_flip == 0:
            di_g.add_edge(e[0], e[1])
        else:
            di_g.add_edge(e[1], e[0])
    strong_connected = nx.is_strongly_connected(di_g)    

    while not strong_connected:
        for e in di_g.edges():
            if (e[1], e[0]) not in di_g.edges():
                di_g.add_edge(e[1], e[0])
        strong_connected = nx.is_strongly_connected(di_g)
    
    for e in di_g.edges():
        # add random edge weights (capacities) from 15-100
        di_g.edges[e]['weight'] = np.random.randint(15,100)

    sorted_nodes = sorted(di_g.degree, key=lambda x: x[1], reverse=True)
    src_node = sorted_nodes[0][0]
    tgt_node = sorted_nodes[1][0]

    file_to_write = open(f'./testcases/{n}_{k}.2dnnflow', 'w')
    file_to_write.write(f"{nx.number_of_nodes(di_g)}\n")
    file_to_write.write(f"{src_node} {tgt_node}\n")
    for e in di_g.edges():
        file_to_write.write(f"{e[0]} {e[1]} {di_g[e[0]][e[1]]['weight']}\n")
    file_to_write.close()


def create_random_erdos_renyi_graphs(n, p, seed=42):
    g = nx.gnp_random_graph(n, p, seed=seed, directed=False)

    while nx.is_connected(g) == False:
        seed += 1
        g = nx.gnp_random_graph(n, p, seed=seed, directed=False)
    # print(f'The seed is {seed}')

    di_g = nx.DiGraph()
    for e in g.edges():
        coin_flip = random.randint(0, 1)
        if coin_flip == 0:
            di_g.add_edge(e[0], e[1])
        else:
            di_g.add_edge(e[1], e[0])

    strong_connected = nx.is_strongly_connected(di_g)   
    while not strong_connected:
        for e in di_g.edges():
            if (e[1], e[0]) not in di_g.edges():
                di_g.add_edge(e[1], e[0])
        strong_connected = nx.is_strongly_connected(di_g)

    for e in di_g.edges():
        # add random edge weights (capacities) from 15-100
        di_g.edges[e]['weight'] = np.random.randint(15,100)
    
    sorted_nodes = sorted(di_g.degree, key=lambda x: x[1], reverse=True)
    src_node = sorted_nodes[0][0]
    tgt_node = sorted_nodes[1][0]

    file_to_write = open(f'./testcases/{n}_{p}.erdosflow', 'w')
    file_to_write.write(f"{nx.number_of_nodes(di_g)}\n")
    file_to_write.write(f"{src_node} {tgt_node}\n")
    for e in di_g.edges():
        file_to_write.write(f"{e[0]} {e[1]} {di_g[e[0]][e[1]]['weight']}\n")
    file_to_write.close()

# add the knapsack constraint to objective
def solve_nip_gurobi_relaxation(G, src_node, tgt_node, B, λ_min, λ_max):
    def compute_objective(di_g, x):
        di_g_copy = di_g.copy()
        edge_counter = 0
        for (u,v) in di_g.edges():
            if x[edge_counter] == 1:
                di_g_copy.remove_edge(u,v)
            edge_counter += 1

        flow_value, flow_dict = nx.maximum_flow(di_g_copy, src_node, tgt_node, capacity='weight')
        return flow_value
    
    finished = False
    best_obj = np.inf
    best_sol = []
    # previous_iteration_feasible = False
    list_of_λs = []
    num_of_iterations_without_improvement = 0
    curr_ite = 0


    while not finished and num_of_iterations_without_improvement < 5:
        curr_ite += 1
        curr_λ = round((λ_min + λ_max)/2, 2)
        list_of_λs.append(curr_λ)

        if len(list_of_λs) > 5:
            list_of_λs.pop(0)

        m = Model("NIP")
        m.setParam('OutputFlag', 0)

        # Create binary variables x indexed over edges of G
        x = m.addVars(G.edges(), vtype=GRB.BINARY, name='x')

        # Create binary variables gamma indexed over edges of G
        gamma = m.addVars(G.edges(), vtype=GRB.BINARY, name='gamma')

        # Create binary variables pi indexed over nodes of G
        pi = m.addVars(G.nodes(), vtype=GRB.BINARY, name='pi')

        # Set objective
        m.setObjective(quicksum(G[u][v]['weight']*(1-x[u,v])*gamma[u,v] for u,v in G.edges()) + curr_λ*(quicksum(x[u,v] for u,v in G.edges()) - B), GRB.MINIMIZE)

        # Add constraints
        # For each edge (i,j) pi_i - pi_j - gamma_ij <= 0
        for u,v in G.edges():
            m.addConstr(pi[u] - pi[v] - gamma[u,v] <= 0)

        # pi_t - pi_s <= -1
        m.addConstr(pi[tgt_node] - pi[src_node] <= -1)

        # Solve the model
        m.optimize()

        x_vals = []
        for e in G.edges():
            x_vals.append(x[e].X)
        
        num_interdicted_edge = sum(x_vals)
        curr_obj = compute_objective(G, x_vals)

        print(f'Iteration {curr_ite}, Current lambda is {curr_λ}, Number of interdicted edges is {num_interdicted_edge}, Current objective is {curr_obj}')
        # print(f'Current lambda is {curr_λ}')
        # print(f'Number of interdicted edges is {num_interdicted_edge}')
        # print(f'Current objective is {curr_obj}')
        # print('-------------------')

        # if num_interdicted_edge <= B: # we can decrease the lambda and see if we get better obj
        #     λ_max = curr_λ #discard the right half of the lambda interval
        #     if previous_iteration_feasible == False:
        #         previous_iteration_feasible = True
        #         best_obj = curr_obj
        #     else: # the previous iteration already has a feasible solution, we need to check if we can improve it further
        #         if curr_obj >= best_obj:
        #             # finished = True # we return the previous feasible solution
        #             num_of_iterations_without_improvement += 1
        #             continue
        #         else:
        #             # keep iterating until we cannot improve the solution further
        #             num_of_iterations_without_improvement = 0 # reset counter
        #             best_obj = curr_obj
        # else:
        #     num_of_iterations_without_improvement = 0
        #     λ_min = curr_λ #discard the left half of the lambda interval
        #     if previous_iteration_feasible == True:
        #         previous_iteration_feasible = False
        #         continue
        #     elif best_obj < np.inf and previous_iteration_feasible == False:
        #         finished = True

        if num_interdicted_edge <= B: # we can decrease the lambda and see if we get better obj
            λ_max = curr_λ #discard the right half of the lambda interval
            if curr_obj >= best_obj:
                num_of_iterations_without_improvement += 1
            else:
                num_of_iterations_without_improvement = 0 # reset counter
                best_obj = curr_obj
        else:
            num_of_iterations_without_improvement = 0
            λ_min = curr_λ #discard the left half of the lambda interval
        
        if len(list_of_λs) == 5 and np.std(list_of_λs) < 0.1:
            finished = True

    print('-------------------')
    print(f'Best obj is {best_obj}')

# add dual constraint to the objective
def gurobi_verify_parameters(G, src_node, tgt_node, B, λ, β):
    def compute_objective(di_g, x):
        di_g_copy = di_g.copy()
        edge_counter = 0
        for (u,v) in di_g.edges():
            if x[edge_counter] == 1:
                # print(edge_counter)
                di_g_copy.remove_edge(u,v)
            edge_counter += 1

        flow_value, flow_dict = nx.maximum_flow(di_g_copy, src_node, tgt_node, capacity='weight')
        return flow_value
    
    def check_knapsack_constraint(G, gamma, pi):
        total_sum = 0
        for (u,v) in G.edges():
            total_sum += pi[u].x * (1-pi[v].x) * (1-gamma[u,v].x)
        return total_sum

    m = Model("NIP")
    m.setParam('OutputFlag', 0)

    x = m.addVars(G.edges(), vtype=GRB.BINARY, name='x')
    gamma = m.addVars(G.edges(), vtype=GRB.BINARY, name='gamma')
    pi = m.addVars(G.nodes(), vtype=GRB.BINARY, name='pi')
    w = m.addVars(G.edges(), vtype=GRB.BINARY, name='w')


    obj_func = 0
    for (u,v) in G.edges():
        temp_u = pi[u]
        temp_v = pi[v]

        if u == src_node:
            temp_u = 1
        elif u == tgt_node:
            temp_u = 0

        if v == src_node:
            temp_v = 1
        elif v == tgt_node:
            temp_v = 0
        
        obj_func += β * (temp_u - temp_u * temp_v - temp_u * gamma[u,v] + w[u,v]*(temp_u + temp_v + gamma[u,v]-1) + (temp_u * temp_v + temp_u * gamma[u,v] + temp_v * gamma[u,v]) - (temp_u + temp_v + gamma[u,v]))

    # Set objective
    # m.setObjective(quicksum(G[u][v]['weight']*(1-x[u,v])*gamma[u,v] for u,v in G.edges()) + curr_β*quicksum(pi[u] - pi[v]*pi[u] - pi[u]*gamma[u,v] + w[u,v]*(pi[u] + pi[v] + gamma[u,v]-1) + (pi[u] * pi[v] + pi[u] * gamma[u,v] + pi[v] * gamma[u,v]) - (pi[u] + pi[v] + gamma[u,v]) for u,v in G.edges()), GRB.MINIMIZE)

    # m.setObjective(quicksum(G[u][v]['weight']*(1-x[u,v])*gamma[u,v] for u,v in G.edges()) + curr_β*quicksum(pi[u] - pi[u]*pi[v] - pi[u]*gamma[u,v] + w[u,v]*(pi[u] + pi[v] + gamma[u,v]-1) + (pi[u] * pi[v] + pi[u] * gamma[u,v] + pi[v] * gamma[u,v]) - (pi[u] + pi[v] + gamma[u,v]) for u,v in G.edges()) + 19.25*(quicksum(x[u,v] for u,v in G.edges()) - B), GRB.MINIMIZE)

    m.setObjective(quicksum(G[u][v]['weight']*(1-x[u,v])*gamma[u,v] for u,v in G.edges()) + obj_func + λ * (quicksum(x[u,v] for u,v in G.edges()) - B), GRB.MINIMIZE)

    # pi_t - pi_s <= -1
    # m.addConstr(pi[tgt_node] - pi[src_node] <= -1)

    # m.addConstr(quicksum(x[u,v] for u,v in G.edges()) <= B)

    # Solve the model
    m.optimize()
    x_vals = []
    interdicted_edges = []
    for e in G.edges():
        x_vals.append(x[e].X)
        if x[e].x == 1.0:
            interdicted_edges.append(e)

    num_interdicted_edge = sum(x_vals)
    curr_obj = compute_objective(G, x_vals)
        
    print('----------------------------')
    print(f'NIP Obj is {curr_obj}, Gurobi Objective is {m.ObjVal}')
    print(f"Interdicted edges are: {interdicted_edges}")
    print('----------------------------')
    
# solve the original NIP MIP
def solve_nip_gurobi(G, src_node, tgt_node, B):
    m = Model("NIP")

    m.setParam('OutputFlag', 0)
    # m.setParam('PoolSolutions', 5)

    # Create binary variables x indexed over edges of G
    x = m.addVars(G.edges(), vtype=GRB.BINARY, name='x')

    # Create binary variables gamma indexed over edges of G
    gamma = m.addVars(G.edges(), vtype=GRB.BINARY, name='gamma')

    # Create binary variables pi indexed over nodes of G
    pi = m.addVars(G.nodes(), vtype=GRB.BINARY, name='pi')

    # Set objective
    # Minimize sum over edges of weights times (1-x) times gamma
    m.setObjective(quicksum(G[u][v]['weight']*(1-x[u,v])*gamma[u,v] for u,v in G.edges()), GRB.MINIMIZE)

    # Add constraints
    # For each edge (i,j) pi_i - pi_j - gamma_ij <= 0
    for u,v in G.edges():
        m.addConstr(pi[u] - pi[v] - gamma[u,v] <= 0)

    # Equality knapsack sontraint over x in all edges being less than B
    m.addConstr(quicksum(x[u,v] for u,v in G.edges()) <= B)

    # pi_t - pi_s <= -1
    m.addConstr(pi[tgt_node] - pi[src_node] <= -1)

    # Solve the model
    m.optimize()

    x_vals = []
    interdicted_edges = []
    for e in G.edges():
        x_vals.append(x[e].X)
        if x[e].x == 1.0:
            interdicted_edges.append(e)

    print('----------------------------')
    print(f"Optimal Obj Val from Gurobi is {m.ObjVal}")
    print(f"Interdicted edges are: {interdicted_edges}")
    print('----------------------------')

def improve_dwave_solution(di_g, sample, src_node, tgt_node, B):
    def decompose_dwave_sample(sample):
        num_edges = len(di_g.edges())
        num_nodes = len(di_g.nodes())
        x = np.zeros(num_edges, dtype=int)
        γ = np.zeros(num_edges, dtype=int)
        π = np.zeros(num_nodes, dtype=int)

        for i in range(num_edges):
            if i not in sample:
                x[i] = 0
            else:
                x[i] = sample[i]

            if i+num_edges not in sample:
                # print(i)
                γ[i] = 0
            else:
                γ[i] = sample[i+num_edges]

        for i in range(num_nodes):
            if i+2*num_edges not in sample:
                π[i] = 0
            else:
                π[i] = sample[i+2*num_edges]
            
            if i == src_node:
                π[i] = 1
            elif i == tgt_node:
                π[i] = 0
        
        return x, γ, π
    
    def compute_objective(di_g, src_node, tgt_node, interdicted_edges):
        di_g_copy = di_g.copy()
        for (u,v) in interdicted_edges:
            di_g_copy.remove_edge(u,v)

        flow_value, flow_dict = nx.maximum_flow(di_g_copy, src_node, tgt_node, capacity='weight')
        return flow_value
    
    def compute_dual_constraint(di_g, π, γ):
        edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
        constraint = 0
        edge_counter = 0
        for (u,v) in edge_list:
            constraint += π[u]*(1-π[v])*(1-γ[edge_counter])
            # if π[u]*(1-π[v])*(1-γ[edge_counter]) != 0:
            #     print(u,v, edge_counter)
            #     print(π[u], π[v], γ[edge_counter])
                
            edge_counter += 1
        return constraint
    
    x, γ, π = decompose_dwave_sample(sample)

    edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
    interdicted_edges = []
    for i in range(len(x)):
        if x[i] == 1:
            interdicted_edges.append(edge_list[i])

    # greedily interdict edge
    if sum(x) < B:
        for _ in range(B - sum(x)):
            best_obj = np.inf
            best_candidate_interdicted_edge = (0,0)

            for i in range(len(edge_list)):
                candidate_interdicted_edge = edge_list[i]
                if candidate_interdicted_edge not in interdicted_edges:
                    temp_interdicted_edges = interdicted_edges.copy()
                    temp_interdicted_edges.append(candidate_interdicted_edge)
                    temp_obj = compute_objective(di_g, src_node, tgt_node,  temp_interdicted_edges)
                    if temp_obj < best_obj:
                        best_obj = temp_obj
                        best_candidate_interdicted_edge = candidate_interdicted_edge
            interdicted_edges.append(best_candidate_interdicted_edge)
    
    # greddily un-interdict edge
    elif sum(x) > B:
        for _ in range(sum(x) - B):
            best_obj = np.inf
            best_candidate_interdicted_edge = (0,0)

            for e in interdicted_edges:
                temp_interdicted_edges = interdicted_edges.copy()
                temp_interdicted_edges.remove(e)
                temp_obj = compute_objective(di_g, src_node, tgt_node,  temp_interdicted_edges)
                if temp_obj < best_obj:
                    best_obj = temp_obj
                    best_candidate_interdicted_edge = e

            interdicted_edges.remove(best_candidate_interdicted_edge)

    # dual_constraint = compute_dual_constraint(di_g, π, γ)
    # if dual_constraint != 0:
    #     pass

    return interdicted_edges, compute_objective(di_g, src_node, tgt_node,  interdicted_edges)

def solve_nip_qubo(name, di_g, λ_min, λ_max, src_node, tgt_node, B=2, solver='sa'):
    # decompose qubo samples to variables in our integer program, note: the order of edges is VERY important
    def decompose_dwave_sample(sample):
        num_edges = len(di_g.edges())
        num_nodes = len(di_g.nodes())
        x = np.zeros(num_edges, dtype=int)
        γ = np.zeros(num_edges, dtype=int)
        π = np.zeros(num_nodes, dtype=int)

        for i in range(num_edges):
            if i not in sample:
                x[i] = 0
            else:
                x[i] = sample[i]

            if i+num_edges not in sample:
                print(i)
                γ[i] = 0
            else:
                γ[i] = sample[i+num_edges]

        for i in range(num_nodes):
            if i+2*num_edges not in sample:
                π[i] = 0
            else:
                π[i] = sample[i+2*num_edges]
            
            if i == src_node:
                π[i] = 1
            elif i == tgt_node:
                π[i] = 0
        
        return x, γ, π
    
    # compute the objective, \sum u_e * (1-x_e) * gamma_e, we want to minimize this
    def compute_objective(di_g, x, γ):
        edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
        di_g_copy = di_g.copy()
        edge_counter = 0
        for (u,v) in edge_list:
            if x[edge_counter] == 1:
                di_g_copy.remove_edge(u,v)
            edge_counter += 1

        flow_value, flow_dict = nx.maximum_flow(di_g_copy, src_node, tgt_node, capacity='weight')
        return flow_value
        # edge_counter = 0
        # obj = 0
        # for (u,v) in di_g.edges():
        #     obj += di_g[u][v]['weight'] * (1 - x[edge_counter]) * γ[edge_counter]
        #     edge_counter += 1
        # return obj

    # compute the dual constraint pi_u * (1 - pi_v) * (1 - gamma_e), this value should be 0
    def compute_dual_constraint(di_g, π, γ):
        edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
        constraint = 0
        edge_counter = 0
        for (u,v) in edge_list:
            constraint += π[u]*(1-π[v])*(1-γ[edge_counter])
            # if π[u]*(1-π[v])*(1-γ[edge_counter]) != 0:
            #     print(u,v, edge_counter)
            #     print(π[u], π[v], γ[edge_counter])
                
            edge_counter += 1
        return constraint

    # edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
    finished = False 
    best_obj = np.inf
    best_sol = []
    # previous_iteration_feasible = False
    list_of_λs = []
    num_of_iterations_without_improvement = 0
    
    while not finished:
        curr_λ = round((λ_min + λ_max) / 2, 2)
        # curr_λ = 28.0
        list_of_λs.append(curr_λ)

        if len(list_of_λs) > 5:
            list_of_λs.pop(0)

        print(f'current lambda is {curr_λ}')
        if os.path.isfile(f"./qubos/{name}_{curr_λ}.qubo") == False:
            subprocess.run(['julia','nip_cubic.jl',name, str(curr_λ), str(B)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        
        num_of_samples_satisfy_cardinality_constraint = 0

        if solver == 'sa':
            f = open(f"./qubos/{name}_{curr_λ}.qubo", 'r')
            lines = f.readlines()
            f.close()
            os.remove(f"./qubos/{name}_{curr_λ}.qubo")

            h = {}
            J = {}
            
            for line in lines:
                if line.startswith('c scale'):
                    words = line.split()
                    scale = float(words[-1])
                    continue
                elif line.startswith('c offset'):
                    words = line.split()
                    offset = float(words[-1])
                    continue
                elif line.startswith('c') or line.startswith('p'):
                    continue

                words = line.split()
                src = int(words[0])
                tgt = int(words[1])
                weight = float(words[2])

                if src == tgt:
                    h[src] = weight
                else:
                    J[(src, tgt)] = weight
            
            bqm = dimod.BinaryQuadraticModel(h, J, offset, dimod.BINARY)
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=2500)
            # curr_best_sample = sampleset.first.sample

            records = sampleset.record
            records = sorted(records, key=lambda x:x[1])
            variable_labels = sampleset.variables
            best_samples = []
            for k in range(3):
                curr_sample = records[k][0]
                currs_sample_dict = {}
                for m in range(len(variable_labels)):
                    currs_sample_dict[variable_labels[m]] = curr_sample[m]
                
                best_samples.append(currs_sample_dict)
            
            for k in range(100):
                curr_sample = records[k][0]
                currs_sample_dict = {}
                for m in range(len(variable_labels)):
                    currs_sample_dict[variable_labels[m]] = curr_sample[m]
                x, γ, π = decompose_dwave_sample(currs_sample_dict)
                if sum(x) <= B:
                    num_of_samples_satisfy_cardinality_constraint += 1

            best_energies = [records[0][1], records[1][1], records[2][1]]

            # Count number of qubits used in D-Wave after minor embedding
            # pegasus = dnx.pegasus_graph(16)
            # label_pairs = J.keys()
            # embedding = minorminer.find_embedding(label_pairs, pegasus)
            # plt.figure(figsize=(32, 32))
            # dnx.draw_pegasus_embedding(pegasus, embedding,node_size=1)
            # plt.savefig('pegasus_embedding.png')
            # total_qubit_used = 0
            # for node in embedding:
            #     total_qubit_used += len(embedding[node])

        elif solver == 'qci':
            curr_best_sample = solve_qubo_via_qci.solve_qubo(f"./qubos/{name}_{curr_λ}.qubo", di_g)

        else:
            print(f"Unsupported solver: {solver}")
            exit(1)

        
        best_sol_interdicted_edge = []
        best_energy = np.inf
        for ite in range(len(best_samples)): 
            print(f'Sample {ite+1}:')
            curr_best_sample = best_samples[ite]
            curr_best_energy = best_energies[ite]

            x, γ, π = decompose_dwave_sample(curr_best_sample)
            print(f"Best energy found: {curr_best_energy}")

            num_interdicted_edge = sum(x)
            edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
            interdicted_edges = []
            for i in range(len(x)):
                if x[i] == 1:
                    interdicted_edges.append(edge_list[i])
        
        
            curr_obj = compute_objective(di_g, x, γ)
            print(f'Number of interdicted edge is {num_interdicted_edge}, penalty is {curr_λ * (num_interdicted_edge - B)}')
            print(f'Interdicted edges are: {interdicted_edges}')
            print(f'Objective is {curr_obj}')
            if compute_dual_constraint(di_g, π, γ) != 0:
                print("Dual constraint is non-zero, check your QUBO!")
                # exit(1)
            print()

            interdicted_edges_improved, objective_improved = improve_dwave_solution(di_g, curr_best_sample, src_node, tgt_node, B)
            print(f'Interdicted edges are greedy improvement: {interdicted_edges_improved}')
            print(f'Objective are greedy improvement is {objective_improved}')
            print('----------------------------')

            if objective_improved < best_energy:
                best_energy = objective_improved
                best_sol_interdicted_edge = interdicted_edges_improved

        # print(f'Dual Constraint is {compute_dual_constraint(di_g, π, γ)}')
        # if num_interdicted_edge <= B: # we can decrease the lambda and see if we get better obj
        #     λ_max = curr_λ #discard the right half of the lambda interval
        #     if previous_iteration_feasible == False:
        #         previous_iteration_feasible = True
        #         best_obj = curr_obj
        #         best_sol = (x, γ, π)
        #     else: # the previous iteration already has a feasible solution, we need to check if we can improve it further
        #         if curr_obj >= best_obj:
        #             finished = True # we return the previous feasible solution
        #             continue
        #         else:
        #             # keep iterating until we cannot improve the solution further
        #             best_obj = curr_obj
        #             best_sol = (x, γ, π)
        # else:
        #     λ_min = curr_λ #discard the left half of the lambda interval
        #     if previous_iteration_feasible == True:
        #         previous_iteration_feasible = False
        #         continue
        #     if best_obj < np.inf and previous_iteration_feasible == False:
        #         break

        if num_of_samples_satisfy_cardinality_constraint > 50: # we can decrease the lambda and see if we get better obj
            λ_max = curr_λ #discard the right half of the lambda interval
            # λ_max *= 0.8
            if curr_obj >= best_obj:
                num_of_iterations_without_improvement += 1
            else:
                num_of_iterations_without_improvement = 0 # reset counter
                best_obj = curr_obj
        else:
            num_of_iterations_without_improvement = 0
            λ_min = curr_λ #discard the left half of the lambda interval
            # λ_min *= 1.2
        
        if len(list_of_λs) == 5 and np.std(list_of_λs) < 0.1:
            finished = True
        
        if num_of_iterations_without_improvement >= 5:
            finished = True

        # num_interdicted_edge = len(best_sol_interdicted_edge)
        # if num_interdicted_edge <= B: # we can decrease the lambda and see if we get better obj
        #     # λ_max = curr_λ #discard the right half of the lambda interval
        #     λ_max *= 0.9
        #     if curr_obj >= best_obj:
        #         num_of_iterations_without_improvement += 1
        #     else:
        #         num_of_iterations_without_improvement = 0 # reset counter
        #         best_obj = curr_obj
        # else:
        #     num_of_iterations_without_improvement = 0
        #     # λ_min = curr_λ #discard the left half of the lambda interval
        #     λ_min *= 1.1
        
        # if len(list_of_λs) == 5 and np.std(list_of_λs) < 0.1:
        #     finished = True

    print('------------------------------------')
    print(f'Best OBJ is {best_obj}')
    print('------------------------------------')

def solve_nip_qubo_repair_once_end(name, di_g, λ_min, λ_max, src_node, tgt_node, B=2, solver='sa'):
    # decompose qubo samples to variables in our integer program, note: the order of edges is VERY important
    def decompose_dwave_sample(sample):
        num_edges = len(di_g.edges())
        num_nodes = len(di_g.nodes())
        x = np.zeros(num_edges, dtype=int)
        γ = np.zeros(num_edges, dtype=int)
        π = np.zeros(num_nodes, dtype=int)

        for i in range(num_edges):
            if i not in sample:
                x[i] = 0
            else:
                x[i] = sample[i]

            if i+num_edges not in sample:
                # print(i)
                γ[i] = 0
            else:
                γ[i] = sample[i+num_edges]

        for i in range(num_nodes):
            if i+2*num_edges not in sample:
                π[i] = 0
            else:
                π[i] = sample[i+2*num_edges]
            
            if i == src_node:
                π[i] = 1
            elif i == tgt_node:
                π[i] = 0
        
        return x, γ, π
    
    # compute the objective, \sum u_e * (1-x_e) * gamma_e, we want to minimize this
    def compute_objective(di_g, x, γ):
        edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
        di_g_copy = di_g.copy()
        edge_counter = 0
        for (u,v) in edge_list:
            if x[edge_counter] == 1:
                di_g_copy.remove_edge(u,v)
            edge_counter += 1

        flow_value, flow_dict = nx.maximum_flow(di_g_copy, src_node, tgt_node, capacity='weight')
        return flow_value
        # edge_counter = 0
        # obj = 0
        # for (u,v) in di_g.edges():
        #     obj += di_g[u][v]['weight'] * (1 - x[edge_counter]) * γ[edge_counter]
        #     edge_counter += 1
        # return obj

    # compute the dual constraint pi_u * (1 - pi_v) * (1 - gamma_e), this value should be 0
    def compute_dual_constraint(di_g, π, γ):
        edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
        constraint = 0
        edge_counter = 0
        for (u,v) in edge_list:
            constraint += π[u]*(1-π[v])*(1-γ[edge_counter])
            # if π[u]*(1-π[v])*(1-γ[edge_counter]) != 0:
            #     print(u,v, edge_counter)
            #     print(π[u], π[v], γ[edge_counter])
                
            edge_counter += 1
        return constraint

    # edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
    finished = False 
    best_obj = np.inf
    best_sol_records = []
    # previous_iteration_feasible = False
    list_of_λs = []
    num_of_iterations_without_improvement = 0
    
    while not finished:
        curr_λ = round((λ_min + λ_max) / 2, 2)
        # curr_λ = 28.0
        list_of_λs.append(curr_λ)

        if len(list_of_λs) > 5:
            list_of_λs.pop(0)

        print(f'current lambda is {curr_λ}')
        if os.path.isfile(f"./qubos/{name}_{curr_λ}.qubo") == False:
            subprocess.run(['julia','nip_cubic.jl',name, str(curr_λ), str(B)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        
        num_of_samples_satisfy_cardinality_constraint = 0

        if solver == 'sa':
            f = open(f"./qubos/{name}_{curr_λ}.qubo", 'r')
            lines = f.readlines()
            f.close()
            os.remove(f"./qubos/{name}_{curr_λ}.qubo")

            h = {}
            J = {}
            
            for line in lines:
                if line.startswith('c scale'):
                    words = line.split()
                    scale = float(words[-1])
                    continue
                elif line.startswith('c offset'):
                    words = line.split()
                    offset = float(words[-1])
                    continue
                elif line.startswith('c') or line.startswith('p'):
                    continue

                words = line.split()
                src = int(words[0])
                tgt = int(words[1])
                weight = float(words[2])

                if src == tgt:
                    h[src] = weight
                else:
                    J[(src, tgt)] = weight
            
            bqm = dimod.BinaryQuadraticModel(h, J, offset, dimod.BINARY)
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=2500)
            curr_best_sample = sampleset.first.sample

            x, γ, π = decompose_dwave_sample(curr_best_sample)
            curr_obj = compute_objective(di_g, x, γ)

            records = sampleset.record
            records = sorted(records, key=lambda x:x[1])
            variable_labels = sampleset.variables
            # best_samples = []
            # for k in range(3):
            #     curr_sample = records[k][0]
            #     currs_sample_dict = {}
            #     for m in range(len(variable_labels)):
            #         currs_sample_dict[variable_labels[m]] = curr_sample[m]
                
            #     best_samples.append(currs_sample_dict)
            # best_energies = [records[0][1], records[1][1], records[2][1]]
            
            for k in range(100):
                curr_sample = records[k][0]
                currs_sample_dict = {}
                for m in range(len(variable_labels)):
                    currs_sample_dict[variable_labels[m]] = curr_sample[m]
                x, γ, π = decompose_dwave_sample(currs_sample_dict)
                if sum(x) <= B:
                    num_of_samples_satisfy_cardinality_constraint += 1

        elif solver == 'qci':
            curr_best_sample = solve_qubo_via_qci.solve_qubo(f"./qubos/{name}_{curr_λ}.qubo", di_g)

        else:
            print(f"Unsupported solver: {solver}")
            exit(1)

        if num_of_samples_satisfy_cardinality_constraint > 50: # we can decrease the lambda and see if we get better obj
            λ_max = curr_λ #discard the right half of the lambda interval
            # λ_max *= 0.8
            if curr_obj >= best_obj:
                num_of_iterations_without_improvement += 1
            else:
                num_of_iterations_without_improvement = 0 # reset counter
                best_obj = curr_obj
                best_sol_records = records
        else:
            num_of_iterations_without_improvement = 0
            λ_min = curr_λ #discard the left half of the lambda interval
            # λ_min *= 1.2
        
        if len(list_of_λs) == 5 and np.std(list_of_λs) < 0.1:
            finished = True
        
        if num_of_iterations_without_improvement >= 5:
            finished = True

    edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
    interdicted_edge_dict = {}

    for k in range(100):
        curr_sample = best_sol_records[k][0]
        currs_sample_dict = {}
        for m in range(len(variable_labels)):
            currs_sample_dict[variable_labels[m]] = curr_sample[m]
        x, γ, π = decompose_dwave_sample(currs_sample_dict)
        for i in range(len(x)):
            if x[i] == 1:
                if edge_list[i] not in interdicted_edge_dict:
                    interdicted_edge_dict[edge_list[i]] = 1
                else:
                    interdicted_edge_dict[edge_list[i]] += 1

    interdicted_edge_dict = dict(sorted(interdicted_edge_dict.items(), key=lambda item: item[1]))
    print(list(interdicted_edge_dict.items())[0:5])

    best_samples = []
    best_energies = []

    for k in range(10):
        curr_sample = best_sol_records[k][0]
        currs_sample_dict = {}
        for m in range(len(variable_labels)):
            currs_sample_dict[variable_labels[m]] = curr_sample[m]
        
        best_samples.append(currs_sample_dict)
        best_energies.append(best_sol_records[k][1])


    best_sol_interdicted_edge = []
    best_energy = np.inf
    for ite in range(len(best_samples)): 
        print(f'Sample {ite+1}:')
        curr_best_sample = best_samples[ite]
        curr_best_energy = best_energies[ite]
        x, γ, π = decompose_dwave_sample(curr_best_sample)
        print(f"Best energy found: {curr_best_energy}")

        num_interdicted_edge = sum(x)
        interdicted_edges = []
        for i in range(len(x)):
            if x[i] == 1:
                interdicted_edges.append(edge_list[i])
    
    
        curr_obj = compute_objective(di_g, x, γ)
        print(f'Number of interdicted edge is {num_interdicted_edge}, penalty is {curr_λ * (num_interdicted_edge - B)}')
        print(f'Interdicted edges are: {interdicted_edges}')
        print(f'Objective is {curr_obj}')
        if compute_dual_constraint(di_g, π, γ) != 0:
            print("Dual constraint is non-zero, check your QUBO!")
            # exit(1)
        print()

        interdicted_edges_improved, objective_improved = improve_dwave_solution(di_g, curr_best_sample, src_node, tgt_node, B)
        print(f'Interdicted edges are greedy improvement: {interdicted_edges_improved}')
        print(f'Objective are greedy improvement is {objective_improved}')
        print('----------------------------')

        if objective_improved < best_energy:
            best_energy = objective_improved
            best_sol_interdicted_edge = interdicted_edges_improved
    
    print(f'Best objective: {best_energy}')
    print(f'Best set of interdicted edges: {best_sol_interdicted_edge}')

    # print('------------------------------------')
    # print(f'Best OBJ is {best_obj}')
    # print('------------------------------------')

def nip_to_qubo(B, di_g, src_node, tgt_node, cardinality_penalty_coeff, dual_penalty_coeff):
    num_nodes = len(di_g.nodes())
    num_edges = len(di_g.edges())

    var_x = []
    var_gamma = []
    var_pi = []
    var_w = []

    for i in range(num_edges):
        var_x.append(Binary(f'x_{i}'))

    for i in range(num_edges):
        var_gamma.append(Binary(f'gamma_{i}'))
        
    for i in range(num_nodes):
        var_pi.append(Binary(f'pi_{i}'))

    for i in range(num_edges):
        var_w.append(Binary(f'w_{i}'))
    
    
    edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))

    obj = cardinality_penalty_coeff * (sum(var_x) - B)

    for i in range(len(edge_list)):
        (u,v) = edge_list[i]
        obj += di_g[u][v]['weight'] * (1 - var_x[i]) * var_gamma[i]

        temp_pi_u = var_pi[u]
        temp_pi_v = var_pi[v]

        if u == src_node:
            temp_pi_u = 1
        elif u == tgt_node:
            temp_pi_u = 0
        
        if v == src_node:
            temp_pi_v = 1
        elif v == tgt_node:
            temp_pi_v = 0
        
        obj += dual_penalty_coeff * (temp_pi_u - temp_pi_u * temp_pi_v - temp_pi_u * var_gamma[i] + var_w[i] * (temp_pi_u + temp_pi_v + var_gamma[i] - 1) + (temp_pi_u * temp_pi_v + temp_pi_u * var_gamma[i] + temp_pi_v * var_gamma[i]) - (temp_pi_u + temp_pi_v + var_gamma[i]))

    model = obj.compile()
    bqm = model.to_bqm()
    return bqm

def qubo_verify_parameters(name, di_g, curr_λ, src_node, tgt_node, B=2, solver='sa'):
    # decompose qubo samples to variables in our integer program, note: the order of edges is VERY important
    def decompose_dwave_sample(sample):
        num_edges = len(di_g.edges())
        num_nodes = len(di_g.nodes())
        x = np.zeros(num_edges, dtype=int)
        γ = np.zeros(num_edges, dtype=int)
        π = np.zeros(num_nodes, dtype=int)

        for i in range(num_edges):
            x[i] = sample[f'x_{i}']
            γ[i] = sample[f'gamma_{i}']

        for i in range(num_nodes):
            if i == src_node:
                π[i] = 1
            elif i == tgt_node:
                π[i] = 0
            else:
                π[i] = sample[f'pi_{i}']

        return x, γ, π
    
    def compute_objective_maxflow(di_g, src_node, tgt_node, interdicted_edges):
        di_g_copy = di_g.copy()
        for (u,v) in interdicted_edges:
            di_g_copy.remove_edge(u,v)

        flow_value, flow_dict = nx.maximum_flow(di_g_copy, src_node, tgt_node, capacity='weight')
        return flow_value
    
    # compute the objective, \sum u_e * (1-x_e) * gamma_e, we want to minimize this
    def compute_objective(di_g, x, γ):
        edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
        di_g_copy = di_g.copy()
        edge_counter = 0
        for (u,v) in edge_list:
            if x[edge_counter] == 1:
                di_g_copy.remove_edge(u,v)
            edge_counter += 1

        flow_value, flow_dict = nx.maximum_flow(di_g_copy, src_node, tgt_node, capacity='weight')
        return flow_value
        # edge_counter = 0
        # obj = 0
        # for (u,v) in di_g.edges():
        #     obj += di_g[u][v]['weight'] * (1 - x[edge_counter]) * γ[edge_counter]
        #     edge_counter += 1
        # return obj

    # compute the dual constraint pi_u * (1 - pi_v) * (1 - gamma_e), this value should be 0
    def compute_dual_constraint(di_g, π, γ):
        edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
        constraint = 0
        edge_counter = 0
        for (u,v) in edge_list:
            constraint += π[u]*(1-π[v])*(1-γ[edge_counter])
            # if π[u]*(1-π[v])*(1-γ[edge_counter]) != 0:
            #     print(u,v, edge_counter)
            #     print(π[u], π[v], γ[edge_counter])
                
            edge_counter += 1
        return constraint

    print(f'current lambda is {curr_λ}\n')
    # if os.path.isfile(f"./qubos/{name}_{curr_λ}.qubo") == False:
    #     subprocess.run(['julia','nip_cubic.jl',name, str(curr_λ), str(B)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    
    if solver == 'sa' or solver == 'qa':
        bqm = nip_to_qubo(B, di_g, src_node, tgt_node, curr_λ, curr_λ+1)
        sampleset = solve_qubo_via_dwave.solve_bqm_dwave(bqm, solver)
        records = sampleset.record
        records = sorted(records, key=lambda x:x[1])
        variable_labels = sampleset.variables
        best_samples = []
        best_energies = []

        for k in range(20):
            curr_sample = records[k][0]
            curr_sample_dict = {}
            for m in range(len(variable_labels)):
                curr_sample_dict[variable_labels[m]] = curr_sample[m]
            best_samples.append(curr_sample_dict)
            best_energies.append(records[k][1])
        edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))

        if False:
            # Here we pick the best 100 samples, collect all interdicted edges, and we sample from these edges based on how frequent they occur
            interdicted_edge_dict = {}
            for k in range(100):
                curr_sample = records[k][0]
                currs_sample_dict = {}
                for m in range(len(variable_labels)):
                    currs_sample_dict[variable_labels[m]] = curr_sample[m]
                x, γ, π = decompose_dwave_sample(currs_sample_dict)
                for i in range(len(x)):
                    if x[i] == 1:
                        if edge_list[i] not in interdicted_edge_dict:
                            interdicted_edge_dict[edge_list[i]] = 1
                        else:
                            interdicted_edge_dict[edge_list[i]] += 1

            interdicted_edge_dict = dict(sorted(interdicted_edge_dict.items(), key=lambda item: item[1], reverse=True))
            interdicted_edge_dict = list(interdicted_edge_dict.items())

            sample_prob = np.zeros(len(interdicted_edge_dict))
            for i in range(len(interdicted_edge_dict)):
                sample_prob[i] = interdicted_edge_dict[i][1]
            
            sample_prob /= np.sum(sample_prob)

            for _ in range(20):
                interdicted_edges_indices = np.random.choice(len(interdicted_edge_dict), B, replace=False, p=sample_prob)
                interdicted_edges = []
                for i in range(B):
                    interdicted_edges.append(interdicted_edge_dict[interdicted_edges_indices[i]][0])
                print(compute_objective_maxflow(di_g, src_node, tgt_node,interdicted_edges))
            print(best_energies)
        
        else:
            best_sol_interdicted_edge = []
            best_sol_after_greedy = np.inf
            best_sol_before_greedy = np.inf

            for ite in range(len(best_samples)): 
                print(f'Sample {ite+1}:')
                curr_best_sample = best_samples[ite]
                curr_energy = best_energies[ite]
                x, γ, π = decompose_dwave_sample(curr_best_sample)
                print(f"Energy for current sample: {curr_energy}")

                num_interdicted_edge = sum(x)
                interdicted_edges = []
                for i in range(len(x)):
                    if x[i] == 1:
                        interdicted_edges.append(edge_list[i])
            
            
                curr_obj = compute_objective(di_g, x, γ)

                if curr_obj < best_sol_before_greedy and num_interdicted_edge <= B:
                    best_sol_before_greedy = curr_obj

                print(f'Number of interdicted edge is {num_interdicted_edge}, penalty is {curr_λ * (num_interdicted_edge - B)}')
                print(f'Interdicted edges are: {interdicted_edges}')
                print(f'Objective is {curr_obj}')
                # if compute_dual_constraint(di_g, π, γ) != 0:
                #     print("Dual constraint is non-zero, check your QUBO!")
                    # exit(1)
                print()

                interdicted_edges_improved, objective_improved = improve_dwave_solution(di_g, curr_best_sample, src_node, tgt_node, B)
                print(f'Interdicted edges are greedy improvement: {interdicted_edges_improved}')
                print(f'Objective after greedy improvement is {objective_improved}')
                print('----------------------------')

                if objective_improved < best_sol_after_greedy:
                    best_sol_after_greedy = objective_improved
                    best_sol_interdicted_edge = interdicted_edges_improved
            
            print(f'Best energy: {best_energies[0]}')
            print(f'Best objective before greedy improvement: {best_sol_before_greedy}')
            print(f'Best objective after greedy improvement: {best_sol_after_greedy}')
            print(f'Best set of interdicted edges: {best_sol_interdicted_edge}')


        # for i in range(len(top_interdicted_edges)-1):
        #     for j in range(i+1, len(top_interdicted_edges)):
        #         print(compute_objective_maxflow(di_g, src_node, tgt_node, [top_interdicted_edges[i][0], top_interdicted_edges[j][0]]))

    elif solver == 'qci':
        curr_best_sample = solve_qubo_via_qci.solver_qubo_dirac1(f"./qubos/{name}_{curr_λ}.qubo", di_g)

    else:
        print(f"Unsupported solver: {solver}")
        exit(1)

    # best_sol_interdicted_edge = []
    # best_energy = np.inf

    # for ite in range(len(best_samples)): 
    #     print(f'Sample {ite+1}:')
    #     curr_best_sample = best_samples[ite]
    #     curr_best_energy = best_energies[ite]

    #     x, γ, π = decompose_dwave_sample(curr_best_sample)
    #     print(f"Best energy found: {curr_best_energy}")

    #     improve_dwave_solution(di_g, curr_best_sample, src_node, tgt_node, B)

    #     num_interdicted_edge = sum(x)
    #     edge_list = sorted(di_g.edges(), key=lambda element: (element[0], element[1]))
    #     interdicted_edges = []
    #     for i in range(len(x)):
    #         if x[i] == 1:
    #             interdicted_edges.append(edge_list[i])
    
    
    #     curr_obj = compute_objective(di_g, x, γ)
    #     print(f'Number of interdicted edge is {num_interdicted_edge}, penalty is {curr_λ * (num_interdicted_edge - B)}')
    #     print(f'Interdicted edges are: {interdicted_edges}')
    #     print(f'Objective is {curr_obj}')
    #     if compute_dual_constraint(di_g, π, γ) != 0:
    #         print("Dual constraint is non-zero, check your QUBO!")
    #         # exit(1)
    #     print()

    #     interdicted_edges_improved, objective_improved = improve_dwave_solution(di_g, curr_best_sample, src_node, tgt_node, B)
    #     print(f'Interdicted edges are greedy improvement: {interdicted_edges_improved}')
    #     print(f'Objective are greedy improvement is {objective_improved}')
    #     print('----------------------------')

    #     if objective_improved < best_energy:
    #         best_energy = objective_improved
    #         best_sol_interdicted_edge = interdicted_edges_improved

    # print(f'Best objective: {best_energy}')
    # print(f'Best set of interdicted edges: {best_sol_interdicted_edge}')

    # print(f'Dual Constraint is {compute_dual_constraint(di_g, π, γ)}')
    # if num_interdicted_edge <= B: # we can decrease the lambda and see if we get better obj
    #     λ_max = curr_λ #discard the right half of the lambda interval
    #     if previous_iteration_feasible == False:
    #         previous_iteration_feasible = True
    #         best_obj = curr_obj
    #         best_sol = (x, γ, π)
    #     else: # the previous iteration already has a feasible solution, we need to check if we can improve it further
    #         if curr_obj >= best_obj:
    #             finished = True # we return the previous feasible solution
    #             continue
    #         else:
    #             # keep iterating until we cannot improve the solution further
    #             best_obj = curr_obj
    #             best_sol = (x, γ, π)
    # else:
    #     λ_min = curr_λ #discard the left half of the lambda interval
    #     if previous_iteration_feasible == True:
    #         previous_iteration_feasible = False
    #         continue
    #     if best_obj < np.inf and previous_iteration_feasible == False:
    #         break

def main():
    graph_type = 'planar'
    

    if graph_type == 'erdos':
        # Create random Erdos-Renyi graph
        graph_name = 90
        p = 0.1
        if os.path.isfile(f'./testcases/{graph_name}_{p}.erdosflow') == False:
            create_random_erdos_renyi_graphs(graph_name, p)
            di_g,src_node,tgt_node = read_flow_file(f'./testcases/{graph_name}_{p}.erdosflow')
        else:
            di_g,src_node,tgt_node = read_flow_file(f'./testcases/{graph_name}_{p}.erdosflow')
    elif graph_type == 'planar':
        graph_name = 50
        if os.path.isfile(f'./testcases/{graph_name}.planarflow') == False:
            leda_to_flow_file(graph_name)
            di_g,src_node,tgt_node = read_flow_file(f'./testcases/{graph_name}.planarflow')
        else:
            di_g,src_node,tgt_node = read_flow_file(f'./testcases/{graph_name}.planarflow')

    elif graph_type == '3d_nn':
        graph_name = 100
        k = 6
        if os.path.isfile(f'./testcases/{graph_name}_{k}.2dnnflow') == False:
            create_random_nn_graphs(graph_name, k, 42)
            di_g,src_node,tgt_node = read_flow_file(f'./testcases/{graph_name}_{k}.2dnnflow')
        else:
            di_g,src_node,tgt_node = read_flow_file(f'./testcases/{graph_name}_{k}.2dnnflow')

    B = 12
    print(f'This graph has {len(di_g.nodes())} nodes and {len(di_g.edges())} edges')

    # solve NIP exactly via Gurobi
    solve_nip_gurobi(di_g, src_node, tgt_node, B)

    # solve NIP by first converting it to QUBO, then simulated annealing (dwave-neal)
    λ_max = max(dict(di_g.edges).items(), key=lambda x: x[1]['weight'])[1]['weight']
    λ_min = min(dict(di_g.edges).items(), key=lambda x: x[1]['weight'])[1]['weight']

    # solve NIP via iterative approach using Gurobi
    solve_nip_gurobi_relaxation(di_g, src_node, tgt_node, B, λ_min, λ_max)

    # Verify lambda_1 and lambda_2 using Gurobi 
    test_lambda =36.0
    gurobi_verify_parameters(di_g, src_node, tgt_node, B, test_lambda, test_lambda + 1)
    qubo_verify_parameters(str(graph_name), di_g, test_lambda, src_node, tgt_node, B, solver='sa')

    # solve NIP via iterative approach using QUBO solvers, e.g., neal, qci
    # solve_nip_qubo(graph_name, di_g, λ_min, λ_max, src_node, tgt_node, B)
    # solve_nip_qubo_repair_once_end(graph_name, di_g, λ_min, λ_max, src_node, tgt_node, B)

    flow_value, flow_dict = nx.maximum_flow(di_g, src_node, tgt_node, capacity='weight')
    print(flow_value)
    
if __name__ == "__main__":
    main()