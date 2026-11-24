import time
from amplpy import AMPL, Environment, DataFrame


def normalize(cycle):
    k = cycle.index(min(cycle))
    return cycle[k:] + cycle[:k]


def all_cycles(cycles, path, node, tovisit, adj, K):
    for i in adj[node]:
        if i in path:
            j = path.index(i)
            cycle = normalize(path[j:] + [node])
            cycles.add(tuple(cycle))
        if i in tovisit:
            if K - 1 > 0:
                all_cycles(cycles, path + [node], i, tovisit - {i}, adj, K - 1)
    return cycles


def get_all_cycles(adj, K):
    tovisit = set(adj.keys())
    cycles = set()
    for i in sorted(tovisit):
        tmpvisit = {v for v in tovisit if v > i}
        all_cycles(cycles, [], i, tmpvisit, adj, K)
    return cycles


def kep_cycle(adj, cycles, weights=None, min_tx=0):
    """
    Solve the KEP cycle-packing problem via AMPL/Gurobi.

    :param adj:     adjacency dict  {node: [neighbours]}
    :param cycles:  list of cycle tuples (as returned by get_all_cycles)
    :param weights: optional list of per-cycle weights (same order as cycles).
                    When None the AMPL model falls back to card(CYCLE[c]),
                    i.e. the number of transplants.
    :param min_tx:  minimum number of transplants the solution must contain.
    :return: (objective_value, list_of_selected_cycle_tuples)
    """
    ampl = AMPL()
    ampl.option['solver'] = 'gurobi'
    ampl.option['gurobi_options'] = 'threads=1'
    ampl.read("kep.mod")
    ampl.set['V'] = list(adj)   # vertices are keys in 'adj'
    ampl.param['m'] = len(cycles)
    for i, c in enumerate(cycles):
        ampl.set['CYCLE'][i] = list(c)
    if weights is not None:
        ampl.param['w'] = {i: float(w) for i, w in enumerate(weights)}
    if min_tx > 0:
        ampl.param['min_tx'] = int(min_tx)

    ampl.solve()
    solve_result = ampl.getValue('solve_result')
    assert solve_result == "solved", f"AMPL solve_result: {solve_result}"

    z = ampl.obj['z']
    x = ampl.var['x']
    sol = []
    for i, c in enumerate(cycles):
        if x[i].value() > 0.5:
            sol.append(c)
    return z.value(), sol



if __name__ == "__main__":
    edges = [(1,2), (1,5), (1,8),
             (2,3), (2,4), (2,5), (2,6), (2,8),
             (3,1), (3,2),
             (4,1), (4,8),
             (5,2), (5,3), (5,7),
             (6,2), (6,7), (6,8),
             (7,5), (7,6),
             (8,2), (8,6),
             ]
    
    adj = {}
    for (i,j) in edges:
        if i in adj:
            adj[i].append(j)
        else:
            adj[i] = [j]
        
    cycles = get_all_cycles(adj,3)
    for c in cycles:
        print(c)
    print(len(cycles), "cycles", time.process_time(), "s")
    z, sol = kep_cycle(adj,cycles)
    print("solution:", z, sol)
        

