from pyz3_utils import MySolver, run_query
from z3 import And, Not, Or, If

from .config import ModelConfig
from .model import make_solver
from .plot import plot_model
from .utils import make_periodic


def bbr_low_util(tsteps=10, timeout=10):
    '''Finds an example trace where BBR has < 10% utilization. It can be made
    arbitrarily small, since BBR can get arbitrarily small throughput in our
    model.

    You can simplify the solution somewhat by setting simplify=True, but that
    can cause small numerical errors which makes the solution inconsistent. See
    README for details.

    '''
    c = ModelConfig.default()
    c.T = tsteps
    c.compose = True
    c.cca = "bbr"
    # Simplification isn't necessary, but makes the output a bit easier to
    # understand
    c.simplify = False
    s, v = make_solver(c)
    # Consider the no loss case for simplicity
    s.add(v.L[0] == 0)
    # Ask for < 10% utilization. Can be made arbitrarily small
    s.add(v.S[-1] - v.S[0] < 0.1 * c.C * c.T)
    make_periodic(c, s, v, 2 * c.R)
    qres = run_query(c, s, v, timeout)
    print(qres.satisfiable)
    if str(qres.satisfiable) == "sat":
        plot_model(qres.model, c, qres.v)


def bbr_test(timeout=10):
    c = ModelConfig.default()
    c.compose = True
    c.cca = "bbr"
    c.buf_min = 0.5
    c.buf_max = 0.5
    c.T = 8
    # Simplification isn't necessary, but makes the output a bit easier to
    # understand
    c.simplify = False
    s, v = make_solver(c)
    # Consider the no loss case for simplicity
    s.add(v.L[0] == 0)
    # Ask for < 10% utilization. Can be made arbitrarily small
    #s.add(v.S[-1] - v.S[0] < 0.1 * c.C * c.T)
    s.add(v.L[-1] - v.L[0] >= 0.5 * (v.S[-1] - v.S[0]))
    s.add(v.A[0] == 0)
    s.add(v.r_f[0][0] < c.C)
    s.add(v.r_f[0][1] < c.C)
    s.add(v.r_f[0][2] < c.C)
    make_periodic(c, s, v, 2 * c.R)
    qres = run_query(c, s, v, timeout)
    print(qres.satisfiable)
    if str(qres.satisfiable) == "sat":
        plot_model(qres.model, c, qres.v)


def copa_low_util(timeout=10):
    '''Finds an example where Copa gets < 10% utilization. This is with the default
    model that composes. If c.compose = False, then CCAC cannot find an example
    where utilization is below 50%. copa_proofs.py proves bounds on Copa's
    performance in the non-composing model. When c.compose = True, Copa can get
    arbitrarily low throughput

    '''
    c = ModelConfig.default()
    c.compose = True
    c.cca = "copa"
    c.simplify = False
    c.calculate_qdel = True
    c.unsat_core = True
    c.T = 10
    c.c_initial = 10000
    c.w_min = 100000
    c.l_min = 1000
    s, v = make_solver(c)
    # Consider the no loss case for simplicity
    s.add(v.L[0] == v.L[-1])
    # 10% utilization. Can be made arbitrarily small
    s.add(v.S[-1] - v.S[0] < 0.1 * c.C * c.T)
    make_periodic(c, s, v, c.R + c.D)

    print(s.to_smt2(), file = open("/tmp/ccac.smt2", "w"))
    s.check()
    print(s.statistics())
    qres = run_query(c, s, v, timeout)
    print(qres.satisfiable)
    if str(qres.satisfiable) == "sat":
        plot_model(qres.model, c, qres.v)


def aimd_premature_loss(timeout=60):
    '''Finds a case where AIMD bursts 2 BDP packets where buffer size = 2 BDP and
    cwnd <= 2 BDP. Here 1BDP is due to an ack burst and another BDP is because
    AIMD just detected 1BDP of loss. This analysis created the example
    discussed in section 6 of the paper. As a result, cwnd can reduce to 1 BDP
    even when buffer size is 2BDP, whereas in a fluid model it never goes below
    1.5 BDP.

    '''
    c = ModelConfig.default()
    c.cca = "aimd"
    c.buf_min = 2
    c.buf_max = 2
    c.simplify = False
    c.T = 5

    s, v = make_solver(c)

    # Start with zero loss and zero queue, so CCAC is forced to generate an
    # example trace *from scratch* showing how bad behavior can happen in a
    # network that was perfectly normal to begin with
    s.add(v.L[0] == 0)
    # Restrict alpha to small values, otherwise CCAC can output obvious and
    # uninteresting behavior
    s.add(v.alpha <= 0.1 * c.C * c.R)

    # Does there exist a time where loss happened while cwnd <= 1?
    conds = []
    for t in range(2, c.T - 1):
        conds.append(
            And(
                v.c_f[0][t] <= 2,
                v.Ld_f[0][t + 1] - v.Ld_f[0][t] >=
                1,  # Burst due to loss detection
                v.S[t + 1 - c.R] - v.S[t - c.R] >=
                c.C + 1,  # Burst of BDP acks
                v.A[t + 1] >=
                v.A[t] + 2,  # Sum of the two bursts
                v.L[t+1] > v.L[t]
            ))

    # We don't want an example with timeouts
    for t in range(c.T):
        s.add(Not(v.timeout_f[0][t]))

    s.add(Or(*conds))

    qres = run_query(c, s, v, timeout)
    print(qres.satisfiable)
    if str(qres.satisfiable) == "sat":
        plot_model(qres.model, c, qres.v)


def rocc_high_util(tsteps=10, timeout=10):
    c = ModelConfig.default()
    c.compose = True
    c.C = 100
    c.cca = "paced"
    c.simplify = False
    c.calculate_qdel = False
    c.unsat_core = True
    c.T = tsteps
    s, v = make_solver(c)
    # Consider the no loss case for simplicity
    s.add(v.L[0] == v.L[-1])
    s.add(v.Ld_f[0][0] == v.L[0])
    first = 4

    # CCA:
    assert first >= c.R
    for t in range(first, c.T):
        next = (v.S_f[0][t-c.R] - v.S_f[0][t-first]) + c.C/100
        s.add(v.c_f[0][t] == If(next > c.C/100, next, c.C/100))

    # High util or cwnd increases
    desired = Or(
        v.S[c.T-1] - v.S[first] >= 0.1 * c.C * (c.T-1-first+1-c.D),
        v.c_f[0][-1] > v.c_f[0][first])
    s.add(Not(desired))
    # make_periodic(c, s, v, c.R + c.D)

    print(s.to_smt2(), file = open("/tmp/ccac.smt2", "w"))
    s.check()
    # print(s.statistics())
    qres = run_query(c, s, v, timeout)
    print(qres.satisfiable)
    if str(qres.satisfiable) == "sat":
        plot_model(qres.model, c, qres.v)



if __name__ == "__main__":

    import time
    for tsteps in range(10, 21, 1):
    # for tsteps in range(10, 11, 1):
        print(f"tsteps = {tsteps}")
        start = time.time()
        rocc_high_util(tsteps, 3600)
        end = time.time()
        print(f"Time taken: {end - start}")

    import sys
    sys.exit(0)

    funcs = {
        "aimd_premature_loss": aimd_premature_loss,
        "bbr_low_util": bbr_low_util,
        "copa_low_util": copa_low_util
    }
    usage = f"Usage: python3 example_queries.py <{'|'.join(funcs.keys())}>"

    if len(sys.argv) != 2:
        print("Expected exactly one command")
        print(usage)
        exit(1)
    cmd = sys.argv[1]
    if cmd in funcs:
        try:
            funcs[cmd]()
        except Exception:
            import sys
            import traceback

            import ipdb
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)

    else:
        print("Command not recognized")
        print(usage)
