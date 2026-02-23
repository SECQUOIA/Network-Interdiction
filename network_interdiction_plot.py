import pandas
import matplotlib.pyplot as plt
import numpy as np

import dwave_networkx as dnx
import networkx as nx

from networkx.algorithms import isomorphism

# G  = dnx.zephyr_graph(12)
# square_no_periodic = nx.grid_graph(dim=(4,4), periodic=[True,False])

# GM = isomorphism.GraphMatcher(G, square_no_periodic)
# print(GM.subgraph_is_isomorphic())
# exit()


csv_file = pandas.read_csv('result_dwave_planar.csv', comment='#')

col_V = list(csv_file['V'])
col_E = list(csv_file['E'])
col_B = list(csv_file['B'])

col_opt_nip = list(csv_file['Opt_NIP'])
col_opt_qubo = list(csv_file['Opt_QUBO'])

col_sa_nip = list(csv_file['SA_NIP'])
col_sa_nip_greedy = list(csv_file['SA_NIP_Greedy'])
col_sa_qubo = list(csv_file['SA_QUBO'])

col_qa_nip = list(csv_file['QA_NIP(advantage1)'])
col_qa_nip_greedy = list(csv_file['QA_NIP(advantage1)_Greedy'])
col_qa_qubo = list(csv_file['QA_QUBO(advantage1)'])

col_maxflow = list(csv_file['maxflow'])

num_entries = len(col_V)
num_vars = []
x_labels = []
flow_reduction_sa = []
flow_reduction_sa_greedy = []
flow_reduction_qa = []
flow_reduction_qa_greedy = []
flow_reduction_opt = []
sa_energy = []
qa_energy = []

for i in range(num_entries):
    num_vars.append(col_V[i] + 3*col_E[i])
    x_labels.append(f"{num_vars[-1]} ({col_B[i]})")
    curr_flow_reduction_opt = (col_maxflow[i] - col_opt_nip[i]) / col_maxflow[i]
    if col_sa_nip[i] == 0:
        flow_reduction_sa.append(0)
    else:
        flow_reduction_sa.append((col_maxflow[i] - col_sa_nip[i]) / col_maxflow[i] / curr_flow_reduction_opt)
    flow_reduction_sa_greedy.append((col_maxflow[i] - col_sa_nip_greedy[i]) / col_maxflow[i] / curr_flow_reduction_opt)
    # flow_reduction_opt.append((col_maxflow[i] - col_opt_nip[i]) / col_maxflow[i])
    if col_qa_nip[i] == 0:
        flow_reduction_qa.append(0)
    else:
        flow_reduction_qa.append((col_maxflow[i] - col_qa_nip[i]) / col_maxflow[i] / curr_flow_reduction_opt)
    flow_reduction_qa_greedy.append((col_maxflow[i] - col_qa_nip_greedy[i]) / col_maxflow[i] / curr_flow_reduction_opt)

    sa_energy.append(col_sa_qubo[i] / col_opt_qubo[i])
    qa_energy.append(col_qa_qubo[i] / col_opt_qubo[i])

    # relative_error.append(col_opt[i] / col_QUBO[i])

plt.figure(figsize=(12, 8))
plt.tight_layout()
ax1 = plt.subplot(1, 1, 1)
barWidth = 0.25
r1 = np.arange(len(flow_reduction_sa)) * 1.5
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

x = np.arange(len(flow_reduction_sa))
plt.tick_params(labelsize=14)

colors=['#99aab5', '#007aff', "#5ac8fa", "#ff3b30", "#ff2d55", "#2c2f33"]
plt.bar(r1, flow_reduction_qa, width=barWidth, edgecolor='white', label='QA',color=colors[0])
plt.bar(r2, flow_reduction_sa, width=barWidth, edgecolor='white', label='SA',color=colors[1])
plt.bar(r3, flow_reduction_qa_greedy, width=barWidth, edgecolor='white', label='QA-Greedy',color=colors[2])
plt.bar(r4, flow_reduction_sa_greedy, width=barWidth, edgecolor='white', label='SA-Greedy',color=colors[3])

plt.axhline(y=1, color='r', linestyle='dotted')
# plt.bar(r3, flow_reduction_opt, width=barWidth, edgecolor='white', label='OPT',color=colors[2])

# plt.rcParams["figure.figsize"] = (12,8)
# plt.bar(x_labels, relative_error)
# plt.title('Comparison between Optimal Solver and QUBO Solver', fontsize=24)
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('#Variables (#Interdicted Edges)', fontsize=20)
plt.ylabel('Flow Reduction from Maxflow against OPT', fontsize=20)
plt.xticks([1.5 * r + 0.375 for r in range(len(flow_reduction_qa))], 
           x_labels, rotation=45)
plt.legend(prop={'size': 12}, loc='upper left', 
           ncol=4, fancybox=False, shadow=True)
plt.tight_layout()
# plt.ylim((-0.05, 1))
# plt.yticks([0.8,0.85,0.9,0.95,1])
plt.ylim((0.0, 1.1))
plt.savefig('result_dwave_planar_flow_reduction.pdf')
plt.clf()

###

plt.figure(figsize=(12, 8))
plt.tight_layout()
ax1 = plt.subplot(1, 1, 1)
barWidth = 0.25
r1 = np.arange(len(flow_reduction_sa)) * 1.5
r2 = [x + barWidth for x in r1]
x = np.arange(len(flow_reduction_sa))
plt.tick_params(labelsize=14)

colors=['#99aab5', '#007aff', "#5ac8fa", "#ff3b30", "#ff2d55", "#2c2f33"]
plt.bar(r1, qa_energy, width=barWidth, edgecolor='white', label='QA',color=colors[0])
plt.bar(r2, sa_energy, width=barWidth, edgecolor='white', label='SA',color=colors[1])


# plt.rcParams["figure.figsize"] = (12,8)
# plt.bar(x_labels, relative_error)
# plt.title('Comparison between Optimal Solver and QUBO Solver', fontsize=24)
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('#Variables (#Interdicted Edges)', fontsize=20)
plt.ylabel('QUBO Energy Ratio against OPT', fontsize=20)
plt.xticks([1.5 * r + 0.15 for r in range(len(flow_reduction_qa))], 
           x_labels, rotation=45)
# plt.yticks([0.9,0.95,1])
plt.legend(prop={'size': 12}, loc='upper left', 
           ncol=3, fancybox=False, shadow=True)
plt.tight_layout()
plt.ylim((0.9, 1.02))
plt.savefig('result_dwave_planar_qubo_energy.pdf')




