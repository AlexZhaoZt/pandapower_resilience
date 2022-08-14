# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import warnings
from sys import stdout

from pandapower.pypower.add_userfcn import add_userfcn
from pandapower.pypower.ppoption import ppoption
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import csgraph
from copy import deepcopy
import numpy as np

from pandapower.auxiliary import ppException, _clean_up, _add_auxiliary_elements
from pandapower.pypower.idx_bus import VM
from pandapower.pypower.opf import opf
from pandapower.pypower.printpf import printpf
from pandapower.pd2ppc import _pd2ppc
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.results import _copy_results_ppci_to_ppc, init_results, \
    _extract_results
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci
from pandapower.pypower.makeBdc import makeBdc
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pypower.idx_brch import PF, PT, QF, QT
from pandapower.pypower.idx_bus import VA, GS, BUS_TYPE, REF, NONE, PV, PQ
from pandapower.pypower.idx_gen import PG, GEN_BUS, PMAX


class OPFNotConverged(ppException):
    """
    Exception being raised in case optimal powerflow did not converge.
    """
    pass

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

def _optimal_powerflow(net, verbose, suppress_warnings, **kwargs):
    ac = net["_options"]["ac"]
    init = net["_options"]["init"]

    ppopt = ppoption(VERBOSE=verbose, OPF_FLOW_LIM=2, PF_DC=not ac, INIT=init, **kwargs)
    net["OPF_converged"] = False
    net["converged"] = False
    _add_auxiliary_elements(net)
    init_results(net, "opf")

    ppc, ppci = _pd2ppc(net)

    # ppci = _add_slack(ppci) # a deepcopied ppci in return--input net is self._grid

    if not ac:
        ppci["bus"][:, VM] = 1.0
    net["_ppc_opf"] = ppci
    if len(net.dcline) > 0:
        ppci = add_userfcn(ppci, 'formulation', _add_dcline_constraints, args=net)

    if init == "pf":
        ppci = _run_pf_before_opf(net, ppci)
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opf(ppci, ppopt)
    else:
        result = opf(ppci, ppopt)
#    net["_ppc_opf"] = result

    if verbose:
        ppopt['OUT_ALL'] = 1
        printpf(baseMVA=result["baseMVA"], bus=result["bus"], gen=result["gen"], fd=stdout,
                branch=result["branch"],  success=result["success"], et=result["et"], ppopt=ppopt)

    if verbose:
        ppopt['OUT_ALL'] = 1
        printpf(baseMVA=result["baseMVA"], bus=result["bus"], gen=result["gen"], fd=stdout,
                branch=result["branch"],  success=result["success"], et=result["et"], ppopt=ppopt)
    if not result["success"]:
        net["OPF_converged"] = False
        raise OPFNotConverged("Optimal Power Flow did not converge!")

    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    mode = net["_options"]["mode"]
    result = _copy_results_ppci_to_ppc(result, ppc, mode=mode)

#    net["_ppc_opf"] = result
    net["OPF_converged"] = True
    _extract_results(net, result)
    _clean_up(net)


def _add_dcline_constraints(om, net):
    # from numpy import hstack, diag, eye, zeros
    ppc = om.get_ppc()
    ndc = net.dcline.in_service.sum()  ## number of in-service DC lines
    if ndc > 0:
        ng = ppc['gen'].shape[0]  ## number of total gens
        Adc = sparse((ndc, ng))
        gen_lookup = net._pd2ppc_lookups["gen"]

        dcline_gens_from = net.gen.index[-2 * ndc::2]
        dcline_gens_to = net.gen.index[-2 * ndc + 1::2]
        for i, (f, t, loss, active) in enumerate(zip(dcline_gens_from, dcline_gens_to,
                                                     net.dcline.loss_percent.values,
                                                     net.dcline.in_service.values)):
            if active:
                Adc[i, gen_lookup[f]] = 1. + loss / 100
                Adc[i, gen_lookup[t]] = 1.

        ## constraints
        nL0 = -net.dcline.loss_mw.values # absolute losses
        #    L1  = -net.dcline.loss_percent.values * 1e-2 #relative losses
        #    Adc = sparse(hstack([zeros((ndc, ng)), diag(1-L1), eye(ndc)]))

        ## add them to the model
        om = om.add_constraints('dcline', Adc, nL0, nL0, ['Pg'])


def _run_pf_before_opf(net, ppci):
#    net._options["numba"] = True
    net._options["tolerance_mva"] = 1e-8
    net._options["max_iteration"] = 10
    net._options["algorithm"] = "nr"
    return _run_newton_raphson_pf(ppci, net["_options"])
