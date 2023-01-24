"""
Test exact calculation methods with `PreciseDigraph` and `PreciseLineGraph` 
against Gillespie simulation data. 

Authors:
    Kee-Myoung Nam

Last updated:
    1/24/2023
"""
import sys
import numpy as np

# Import pygraph 
sys.path.append('/Users/kmnam/Dropbox/gene-regulation/projects/markov-digraphs')
from pygraph import PreciseDigraph, PreciseLineGraph

#########################################################################
def get_exact_stats(graph, terminal_unbind_lograte, terminal_cleave_lograte):
    upper_exit_prob = graph.get_upper_exit_prob(
        10 ** terminal_unbind_lograte, 10 ** terminal_cleave_lograte
    )
    lower_exit_rate_uncond = graph.get_lower_exit_rate(
        10 ** terminal_unbind_lograte
    )
    lower_exit_rate_cond = graph.get_lower_exit_rate(
        10 ** terminal_unbind_lograte, 10 ** terminal_cleave_lograte
    )
    upper_exit_rate = graph.get_upper_exit_rate(
        10 ** terminal_unbind_lograte, 10 ** terminal_cleave_lograte
    )
    lower_end_to_end_time = graph.get_lower_end_to_end_time()
    upper_end_to_end_time = graph.get_upper_end_to_end_time()

    return np.array([
        upper_exit_prob,
        lower_exit_rate_uncond,
        lower_exit_rate_cond,
        upper_exit_rate,
        lower_end_to_end_time,
        upper_end_to_end_time
    ])

#########################################################################
def simulate(graph, terminal_unbind_lograte, terminal_cleave_lograte,
             length, nsim, max_time, rng):
    # Simulate the line graph to obtain end-to-end times
    first_passage_times = []
    for n in range(nsim):
        sim = graph.simulate('0', max_time, rng.integers(0, 999))
        try:
            idx_first_passage_to_end = next(i for i, s in enumerate(sim) if s[0] == str(length))
            time_to_reach_end = sum(s[1] for s in sim[:idx_first_passage_to_end])
        except StopIteration:
            time_to_reach_end = max_time
        first_passage_times.append(time_to_reach_end)
    upper_end_to_end_time = np.mean(first_passage_times)
    first_passage_times = []
    for n in range(nsim):
        sim = graph.simulate(str(length), max_time, rng.integers(0, 999))
        try:
            idx_first_passage_to_0 = next(i for i, s in enumerate(sim) if s[0] == '0')
            time_to_reach_0 = sum(s[1] for s in sim[:idx_first_passage_to_0])
        except StopIteration:
            time_to_reach_0 = max_time
        first_passage_times.append(time_to_reach_0)
    lower_end_to_end_time = np.mean(first_passage_times)
    print(upper_end_to_end_time)
    print(lower_end_to_end_time)

    # Define a new graph with the terminal vertices 
    graph2 = PreciseDigraph()
    for i in range(length + 1):
        graph2.add_node(str(i))
    for i in range(length):
        graph2.add_edge(str(i), str(i + 1), graph.get_edge_label(str(i), str(i + 1)))
        graph2.add_edge(str(i + 1), str(i), graph.get_edge_label(str(i + 1), str(i)))
    graph2.add_node('empty')
    graph2.add_node('star')
    graph2.add_edge('0', 'empty', 10 ** terminal_unbind_lograte)

    # Simulate the new graph (without upper exit) to obtain lower exit rate 
    first_passage_times = []
    for n in range(nsim):
        sim = graph2.simulate('0', max_time, rng.integers(0, 999))
        first_passage_times.append(sum(s[1] for s in sim))
    lower_exit_rate_uncond = 1.0 / np.mean(first_passage_times)

    # Add upper exit node and simulate the new graph to obtain other exit
    # statistics
    graph2.add_edge(str(length), 'star', 10 ** terminal_cleave_lograte)
    first_passage_times_to_lower = []
    first_passage_times_to_upper = []
    exited_upper = 0
    exited = 0
    while exited < nsim:
        sim = graph2.simulate('0', max_time, rng.integers(0, 999))
        if sim[-1][1] < max_time:    # If the simulation has ended and reached an exit node ...
            if sim[-1][0] == '0':
                first_passage_times_to_lower.append(sum(s[1] for s in sim))
                exited += 1
            elif sim[-1][0] == str(length):
                first_passage_times_to_upper.append(sum(s[1] for s in sim))
                exited_upper += 1
                exited += 1
    upper_exit_prob = exited_upper / nsim
    lower_exit_rate_cond = 1.0 / np.mean(first_passage_times_to_lower)
    upper_exit_rate = 1.0 / np.mean(first_passage_times_to_upper)

    return np.array([
        upper_exit_prob,
        lower_exit_rate_uncond,
        lower_exit_rate_cond,
        upper_exit_rate,
        lower_end_to_end_time,
        upper_end_to_end_time
    ])


#########################################################################
# Define the line graph of length 4 with randomly sampled parameter values
length = 4
graph = PreciseLineGraph(length)
rng = np.random.default_rng(1234567890)
a, b = rng.random() * 2 - 1, rng.random() * 2 - 1
if a < b:
    logb = b
    logbp = a
else:
    logb = a
    logbp = b
a, b = rng.random() * 2 - 1, rng.random() * 2 - 1
if a < b:
    logdp = b
    logd = a
else:
    logdp = a
    logd = b
terminal_unbind_lograte = rng.random() * 2 - 1
terminal_cleave_lograte = rng.random() * 2 - 1
for i in range(length):
    graph.set_edge_labels(i, (10 ** logb, 10 ** logd))
exact_stats = get_exact_stats(graph, terminal_unbind_lograte, terminal_cleave_lograte)
print(exact_stats)
nsim = 10000
max_time = 10000
sim_stats = simulate(
    graph, terminal_unbind_lograte, terminal_cleave_lograte, length, nsim,
    max_time, rng
)
print(sim_stats)
