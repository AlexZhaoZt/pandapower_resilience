# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import numpy as np
from numpy import nan_to_num, array, allclose
from copy import deepcopy
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import csgraph
from pandapower.auxiliary import ppException, _clean_up, _add_auxiliary_elements
from pandapower.build_branch import _calc_trafo_parameter, _calc_trafo3w_parameter
from pandapower.build_gen import _build_gen_ppc
from pandapower.pd2ppc import _pd2ppc, _calc_pq_elements_and_add_on_ppc, _ppc2ppci
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci
from pandapower.pf.run_bfswpf import _run_bfswpf
from pandapower.pf.run_dc_pf import _run_dc_pf
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.pf.runpf_pypower import _runpf_pypower
from pandapower.pypower.bustypes import bustypes
from pandapower.pypower.idx_bus import VM
from pandapower.pypower.makeYbus import makeYbus as makeYbus_pypower
from pandapower.pypower.pfsoln import pfsoln as pfsoln_pypower
from pandapower.results import _extract_results, _copy_results_ppci_to_ppc, init_results, \
    verify_results, _ppci_bus_to_ppc, _ppci_other_to_ppc
from pandapower.pypower.makeBdc import makeBdc
from pandapower.pypower.idx_brch import PF, PT, QF, QT
from pandapower.pypower.idx_bus import VA, GS, BUS_TYPE, REF, NONE, PV, PQ
from pandapower.pypower.idx_gen import PG, GEN_BUS, PMAX
try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class AlgorithmUnknown(ppException):
    """
    Exception being raised in case optimal powerflow did not converge.
    """
    pass


class LoadflowNotConverged(ppException):
    """
    Exception being raised in case loadflow did not converge.
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

def _powerflow(net, **kwargs):
    """
    Gets called by runpp or rundcpp with different arguments.
    """

    # get infos from options
    ac = net["_options"]["ac"]
    algorithm = net["_options"]["algorithm"]

    net["converged"] = False
    net["OPF_converged"] = False
    _add_auxiliary_elements(net)

    if not ac or net["_options"]["init_results"]:
        verify_results(net)
    else:
        init_results(net)
    

    if net["_options"]["voltage_depend_loads"] and algorithm not in ['nr', 'bfsw'] and not (
            allclose(net.load.const_z_percent.values, 0) and
            allclose(net.load.const_i_percent.values, 0)):
        logger.error(("pandapower powerflow does not support voltage depend loads for algorithm "
                      "'%s'!") % algorithm)

    # clear lookups
    net._pd2ppc_lookups = {"bus": array([], dtype=int), "ext_grid": array([], dtype=int),
                           "gen": array([], dtype=int), "branch": array([], dtype=int)}

    # convert pandapower net to ppc
    ppc, ppci = _pd2ppc(net)

    # ppci = _add_slack(ppci)

    # store variables
    net["_ppc"] = ppc

    if "VERBOSE" not in kwargs:
        kwargs["VERBOSE"] = 0

    # ----- run the powerflow -----
    result = _run_pf_algorithm(ppci, net["_options"], **kwargs)
    # read the results (=ppci with results) to net
    _ppci_to_net(result, net)


def _recycled_powerflow(net, **kwargs):
    options = net["_options"]
    options["recycle"] = kwargs.get("recycle", None)
    options["init_vm_pu"] = "results"
    options["init_va_degree"] = "results"
    algorithm = options["algorithm"]
    ac = options["ac"]
    ppci = {"bus": net["_ppc"]["internal"]["bus"],
            "gen": net["_ppc"]["internal"]["gen"],
            "branch": net["_ppc"]["internal"]["branch"],
            "baseMVA": net["_ppc"]["internal"]["baseMVA"],
            "internal": net["_ppc"]["internal"],
            }
    if not ac:
        # DC recycle
        result = _run_dc_pf(ppci)
        _ppci_to_net(result, net)
        return
    if algorithm not in ['nr', 'iwamoto_nr'] and ac:
        raise ValueError("recycle is only available with Newton-Raphson power flow. Choose "
                         "algorithm='nr'")

    recycle = options["recycle"]
    ppc = net["_ppc"]
    ppc["success"] = False
    ppc["iterations"] = 0.
    ppc["et"] = 0.

    if "bus_pq" in recycle and recycle["bus_pq"]:
        # update pq values in bus
        _calc_pq_elements_and_add_on_ppc(net, ppc)

    if "trafo" in recycle and recycle["trafo"]:
        # update trafo in branch and Ybus
        lookup = net._pd2ppc_lookups["branch"]
        if "trafo" in lookup:
            _calc_trafo_parameter(net, ppc)
        if "trafo3w" in lookup:
            _calc_trafo3w_parameter(net, ppc)

    if "gen" in recycle and recycle["gen"]:
        # updates the ppc["gen"] part
        _build_gen_ppc(net, ppc)
        ppc["gen"] = nan_to_num(ppc["gen"])

    ppci = _ppc2ppci(ppc, net, ppci=ppci)
    ppci["internal"] = net["_ppc"]["internal"]
    net["_ppc"] = ppc

    # run the Newton-Raphson power flow
    result = _run_newton_raphson_pf(ppci, options)
    ppc["success"] = ppci["success"]
    ppc["iterations"] = ppci["iterations"]
    ppc["et"] = ppci["et"]
    if options["only_v_results"]:
        _ppci_bus_to_ppc(result, ppc)
        _ppci_other_to_ppc(result, ppc, options["mode"])
        return
    # read the results from  result (==ppci) to net
    _ppci_to_net(result, net)


def _run_pf_algorithm(ppci, options, **kwargs):
    algorithm = options["algorithm"]
    ac = options["ac"]

    if ac:
        _, pv, pq = bustypes(ppci["bus"], ppci["gen"])
        # ----- run the powerflow -----
        if pq.shape[0] == 0 and pv.shape[0] == 0 and not options['distributed_slack']:
            # ommission not correct if distributed slack is used
            result = _bypass_pf_and_set_results(ppci, options)
        elif algorithm == 'bfsw':  # forward/backward sweep power flow algorithm
            result = _run_bfswpf(ppci, options, **kwargs)[0]
        elif algorithm in ['nr', 'iwamoto_nr']:
            result = _run_newton_raphson_pf(ppci, options)
        elif algorithm in ['fdbx', 'fdxb', 'gs']:  # algorithms existing within pypower
            result = _runpf_pypower(ppci, options, **kwargs)[0]
        else:
            raise AlgorithmUnknown("Algorithm {0} is unknown!".format(algorithm))
    else:
        result = _run_dc_pf(ppci)

    return result


def _ppci_to_net(result, net):
    # reads the results from result (== ppci with results) to pandapower net

    mode = net["_options"]["mode"]
    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    ppc = net["_ppc"]
    result = _copy_results_ppci_to_ppc(result, ppc, mode)

    # raise if PF was not successful. If DC -> success is always 1
    if result["success"] != 1:
        _clean_up(net, res=False)
        algorithm = net["_options"]["algorithm"]
        max_iteration = net["_options"]["max_iteration"]
        raise LoadflowNotConverged("Power Flow {0} did not converge after "
                                   "{1} iterations!".format(algorithm, max_iteration))
    else:
        net["_ppc"] = result
        net["converged"] = True

    _extract_results(net, result)
    _clean_up(net)


def _bypass_pf_and_set_results(ppci, options):
    Ybus, Yf, Yt = makeYbus_pypower(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    baseMVA, bus, gen, branch, ref, _, pq, _, _, V0, ref_gens = _get_pf_variables_from_ppci(ppci)
    V = ppci["bus"][:, VM]
    bus, gen, branch = pfsoln_pypower(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, ref_gens)
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    ppci["success"] = True
    ppci["iterations"] = 1
    ppci["et"] = 0
    return ppci
