import re
import gzip
import random

def _output_donor(don):
    age, sex, egfr, sbp, bmi, aa, smoker, relat, abo, weight, hlaA, hlaB, hlaDR, pra = don
    return f"{age},{sex},{egfr},{sbp},{bmi},{smoker},{relat},{abo},{weight},{hlaA},{hlaB},{hlaDR}"


def _output_patient(pat):
    age, sex, egfr, sbp, bmi, aa, smoker, relat, abo, weight, hlaA, hlaB, hlaDR, pra = pat
    return f"{age},{sex},{smoker},{abo},{weight},{hlaA},{hlaB},{hlaDR},{egfr}"


def _write_gzip(filename, txt):
    with gzip.open(filename, "wb") as f:
        f.write(txt.encode("utf-8"))


def _write_txt(filename, txt):
    with open(filename, "w") as f:
        f.write(txt)


def _write(filename, txt):
    if filename.find(".gz") > 0:
        _write_gzip(filename, txt)
    else:
        _write(txt)


def _read(filename):
    if filename.find(".gz") > 0:
        with gzip.open(filename) as f:
            data = iter(f.read().split())
    else:
        with open(filename) as f:
            data = iter(f.read().split())
    return data


def write_hist(filename, hist):
    "write csv with kep historical data"
    nrecs = len(hist)
    txt = f"{nrecs}\n"
    for don, pat, noisy_surv in hist:
        txt += f"{_output_donor(don)},{_output_patient(pat)},{noisy_surv}\n"
    _write(filename, txt)


def write_pool(filename, don, pat):
    "write csv with kep's pool data"
    nvert = len(don)
    txt = f"{nvert}\n"
    for i in don:
        txt += f"{i},{_output_donor(don[i])},{_output_patient(pat[i])}\n"
    _write(filename, txt)


def write_pra(filename, pra):
    "write file with pool patients' sensitisation (vertex id, PRA %)"
    txt = f"{len(pra)}\n"
    for i in pra:
        txt += f"{i}\t{pra[i]}\n"
    _write(filename, txt)

def read_pra(filename):
    "read file with pool patients' sensitisation (vertex id, PRA %)"
    data = iter(_read(filename))
    nvert = int(next(data))
    return {int(next(data)): int(next(data)) for _ in range(nvert)}


def write_hidden(filename, adj, surv):
    "write file with hidden data for the kep pool (for final assessment)"
    write_kep(filename, adj, surv)

def read_hidden(filename):
    "write file with hidden data for the kep pool (for final assessment)"
    return read_kep(filename)


def write_kep(filename, adj, w):
    "write file in the 'standard' kep format"

    nvert = len(adj)
    narcs = sum(len(adj[k]) for k in adj)
    txt = f"{nvert}\t{narcs}\n"
    for i in adj:
        for j in adj[i]:
            txt += f"{i}\t{j}\t{w[i, j]}\n"
    _write(filename, txt)


def read_kep(filename):
    "read file in the 'standard' kep format"
    data = iter(_read(filename))
    nvert = int(next(data))
    narcs = int(next(data))
    adj = {}
    w = {}
    for a in range(narcs):
        i = int(next(data))
        j = int(next(data))
        if i in adj:
            adj[i].append(j)
        else:
            adj[i] = [j]
        w[i,j] = float(next(data))
        assert i>=0 and j>=0
    return adj, w



def get_kep_edges(adj,p_):
    """given a directed KEP graph, return its edges (i.e., its 2-cycles)"""
    V = adj.keys()
    p = {}
    edges = []
    for i in V:
        for j in V:
            if j <= i:
                continue
            if j in adj[i] and i in adj[j]:
                edge = frozenset((i,j))
                edges.append(edge)
                p[edge] = 1 - (1-p_[i])*(1-p_[j])*(1-p_[i,j])*(1-p_[j,i])   # corrected 2024-04-16
    return edges,p


if __name__ == "__main__":
    import sys
    import os

    filename = "DATA/50_01.input.gz"
    adj, w = read_kep(filename)
    print(adj)

    write_kep("tmp.gz", adj, w)
    adj2, w2 = read_kep("tmp.gz")
    assert adj2 == adj
    assert w2 == w

