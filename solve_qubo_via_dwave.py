from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from dwave.samplers import SteepestDescentSolver
import numpy as np
import neal, dimod
from minorminer import find_embedding

def count_qubit_usage(emb):
    count = 0
    for logical_qubit in emb:
        count += len(emb[logical_qubit])
    return count

def solve_bqm_dwave(bqm, annealer_type, seed=None):
    """Sample `bqm` with the QPU ('qa') or simulated annealing ('sa').
    `seed` seeds simulated annealing (ignored for 'qa')."""
    if annealer_type == 'qa':
        # qpu_advantage = DWaveSampler(solver='Advantage2_system1.3')
        qpu_advantage = DWaveSampler(solver='Advantage_system6.4')
        # __, target_edgelist, target_adjacency = qpu_advantage.structure
        # emb = find_embedding(J, target_edgelist, verbose=0)
        # print(f"Number of variables: {len(emb)}, Number of used qubits: {count_qubit_usage(emb)}")
        # sampler = FixedEmbeddingComposite(qpu_advantage, emb)
        sampler = EmbeddingComposite(qpu_advantage)
        sampleset = sampler.sample(bqm, num_reads=2500, annealing_time=50)
        solver_greedy = SteepestDescentSolver()
        sampleset_greedy = solver_greedy.sample(bqm, initial_states=sampleset)  
        return sampleset_greedy
    elif annealer_type == 'sa':
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=2500, seed=seed)
        return sampleset
    else:
        print('Unsupported D-Wave sampler, only supports sa and qa')
        exit(1)


def solve_qubo_dwave(qubojl_output, annealer_type):
    f = open(qubojl_output, 'r')
    lines = f.readlines()
    f.close()

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
    if annealer_type == 'qa':
        qpu_advantage = DWaveSampler(solver='Advantage2_system1.3')
        # qpu_advantage = DWaveSampler(solver='Advantage_system6.4')
        # __, target_edgelist, target_adjacency = qpu_advantage.structure
        # emb = find_embedding(J, target_edgelist, verbose=0)
        # print(f"Number of variables: {len(emb)}, Number of used qubits: {count_qubit_usage(emb)}")
        # sampler = FixedEmbeddingComposite(qpu_advantage, emb)
        sampler = EmbeddingComposite(qpu_advantage)
        sampleset = sampler.sample(bqm, num_reads=2500, annealing_time=50)
        solver_greedy = SteepestDescentSolver()
        sampleset_greedy = solver_greedy.sample(bqm, initial_states=sampleset)  
        return sampleset_greedy
    elif annealer_type == 'sa':
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=2500)
        return sampleset
    else:
        print('Unsupported D-Wave sampler, only supports sa and qa')
        exit(1)

    


