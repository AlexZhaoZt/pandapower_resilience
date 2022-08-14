# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



from time import time
from scipy.sparse import csgraph, csr_matrix
from copy import deepcopy
from numpy import pi, zeros, real, bincount
from numpy import ones, flatnonzero as find
import numpy as np

from pandapower.pypower.idx_brch import PF, PT, QF, QT
from pandapower.pypower.idx_bus import VA, GS, BUS_TYPE, REF, NONE, PV, PQ
from pandapower.pypower.idx_gen import PG, GEN_BUS, PMAX
from pandapower.pypower.dcpf import dcpf
from pandapower.pypower.makeBdc import makeBdc
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci, _store_results_from_pf_in_ppci

def _add_slack(ppci):
    '''
    1. check connectivity
    2. if components>1 (meaning isolated) conduct topology check
    3. topo check: when at least one generator (typically non-renewable) and one load in one island, the island works
    4. if the island works, reassign the slack bus (BUS_TYPE=3); otherwise, set the buses and branches out of service
    5. return the modified variables and let them replace the input ppci
    '''

    ppci = deepcopy(ppci)
    baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, _, refgen = _get_pf_variables_from_ppci(ppci)
    isolated = False
    islands = []
    # convert to B matrix (sparse)
    B, Bf, Pbusinj, Pfinj = makeBdc(bus, branch)
    # check connectivity--return a tuple [NO_islands, components_in_each_island]
    conn_B = csgraph.connected_components(B)
    # (2, array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0]))
    # record the buses connected to each island; if not isolated, islands will contain all buses
    for NO_island in range(conn_B[0]):
        islands.append(np.where(conn_B[1] == NO_island))
    #TODO: add the maximum islands threshold here, raise an error if len(islands)>THRESHOLD
    # print(islands)
    # print(conn_B)
    # [(array([0, 1, 2, 3, 4, 6, 7], dtype=int64),), (array([ 5,  8,  9, 10, 11, 12, 13], dtype=int64),)]
    for i in range(len(islands)):   # check the availability of each island
        if np.any(bus[islands[i][:], 1]==REF):    # if there is a slack bus in this island
            pass
        else:
            i_island = islands[i]
            if np.any(bus[i_island, 1] == PV) and np.any(bus[i_island, 2] > 0):   # generator > 0 and load > 0
                bus_flatten = bus[i_island, 1].flatten()
                ref_avail = np.where(bus_flatten==PV)   # PV buses available to be REF
                new_ref = ref_avail[0][0]   # choose the first PV bus--for customization, more selection conditions will be added
                bus[i_island[0][new_ref], 1] = REF  # reassign the slack bus here
                isolated = True
            else:   # if the island cannot work, let them be out of service
                bus[i_island, 1] = NONE
                # print(f"buses {i_island} will be out of service due to isolation\n")

    ppci["bus"] = bus
    ppci["gen"] = gen
    ppci["branch"] = branch

    return ppci

def _run_dc_pf(ppci, island_mode=False):
    t0 = time()
    if island_mode is True:
        baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, _, refgen = _island_check(ppci)
    else:
        baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, _, refgen = _get_pf_variables_from_ppci(ppci)

    '''
    second column in bus returned by _get_pf_var...
    1: pq
    2: pv
    3: slack
    4: out of service
    '''
    ## initial state
    Va0 = bus[:, VA] * (pi / 180.)

    ## build B matrices and phase shift injections
    B, Bf, Pbusinj, Pfinj = makeBdc(bus, branch)

    ## updates Bbus matrix
    ppci['internal']['Bbus'] = B

    ## compute complex bus power injections [generation - load]
    ## adjusted for phase shifters and real shunts
    Pbus = makeSbus(baseMVA, bus, gen) - Pbusinj - bus[:, GS] / baseMVA

    ## "run" the power flow
    Va = dcpf(B, Pbus, Va0, ref, pv, pq)

    ## update data matrices with solution
    branch[:, [QF, QT]] = zeros((branch.shape[0], 2))
    branch[:, PF] = (Bf * Va + Pfinj) * baseMVA
    branch[:, PT] = -branch[:, PF]
    bus[:, VA] = Va * (180. / pi)
    ## update Pg for slack generators
    ## (note: other gens at ref bus are accounted for in Pbus)
    ##      Pg = Pinj + Pload + Gs
    ##      newPg = oldPg + newPinj - oldPinj

    ## ext_grid (refgen) buses
    refgenbus=gen[refgen, GEN_BUS].astype(int)
    ## number of ext_grids (refgen) at those buses
    ext_grids_bus=bincount(refgenbus)
    gen[refgen, PG] = real(gen[refgen, PG] + (B[refgenbus, :] * Va - Pbus[refgenbus]) * baseMVA / ext_grids_bus[refgenbus])

    # store results from DC powerflow for AC powerflow
    et = time() - t0
    success = True
    iterations = 1
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, iterations, et)
    return ppci

# def _island_check(ppci):
#     '''
#     1. check connectivity
#     2. if components>1 (meaning isolated) conduct topology check
#     3. topo check: when at least one generator (typically non-renewable) and one load in one island, the island works
#     4. if the island works, reassign the slack bus (BUS_TYPE=3); otherwise, set the buses and branches out of service
#     5. return the modified variables and let them replace the input ppci
#     '''

#     ppci = deepcopy(ppci)
#     baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, _, refgen = _get_pf_variables_from_ppci(ppci)

#     islands = []
#     # convert to B matrix (sparse)
#     B, Bf, Pbusinj, Pfinj = makeBdc(bus, branch)
#     # check connectivity--return a tuple [NO_islands, components_in_each_island]
#     conn_B = csgraph.connected_components(B)
#     # (2, array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0]))
#     # record the buses connected to each island; if not isolated, islands will contain all buses
#     for NO_island in range(conn_B[0]):
#         islands.append(np.where(conn_B[1] == NO_island))
#     #TODO: add the maximum islands threshold here, raise an error if len(islands)>THRESHOLD
#     # print(islands)
#     # print(conn_B)
#     # [(array([0, 1, 2, 3, 4, 6, 7], dtype=int64),), (array([ 5,  8,  9, 10, 11, 12, 13], dtype=int64),)]
#     for i in range(len(islands)):   # check the availability of each island
#         if np.any(bus[islands[i][:], 1]==REF):    # if there is a slack bus in this island
#             pass
#         else:
#             i_island = islands[i]
#             if np.any(bus[i_island, 1] == PV) and np.any(bus[i_island, 2] > 0):   # generator > 0 and load > 0
#                 bus_flatten = bus[i_island, 1].flatten()
#                 ref_avail = np.where(bus_flatten==PV)   # PV buses available to be REF
#                 new_ref = ref_avail[0][0]   # choose the first PV bus--for customization, more selection conditions will be added
#                 bus[i_island[0][new_ref], 1] = REF  # reassign the slack bus here
#             else:   # if the island cannot work, let them be out of service
#                 bus[i_island, 1] = NONE
#                 # print(f"buses {i_island} will be out of service due to isolation\n")

#     ref = find((bus[:, BUS_TYPE] == REF))  # ref bus index
#     pv = find((bus[:, BUS_TYPE] == PV))  # PV bus indices
#     pq = find((bus[:, BUS_TYPE] == PQ))  # PQ bus indices
#     refgen = np.array([], dtype=int)

#     for i_bus in ref:   # ref_gen indices
#         temp_refgen = find(gen[:, GEN_BUS] == i_bus)
#         refgen = np.concatenate((refgen, temp_refgen))

#     return baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, _, refgen
