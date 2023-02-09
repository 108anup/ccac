import numpy as np
import argparse
from typing import Optional, Union
import z3


class ModelConfig:
    # Number of flows
    N: int
    # Jitter parameter (in timesteps)
    D: int
    # RTT (in timesteps)
    R: int
    # Number of timesteps
    T: int
    # Link rate
    C: float
    # Packets cannot be dropped below this threshold
    buf_min: Optional[Union[float, z3.ExprRef]]
    # Packets have to be dropped above this threshold
    buf_max: Optional[Union[float, z3.ExprRef]]
    # Number of dupacks before sender declares loss
    dupacks: Optional[float]
    # Congestion control algorithm
    cca: str
    # If false, we'll use a model that is more restrictive but does not compose
    compose: bool
    # Additive increase parameter used by various CCAs
    alpha: Union[float, z3.ArithRef] = 1.0
    # Whether or not to use pacing in various CCA
    pacing: bool
    # If compose is false, wastage can only happen if queue length < epsilon
    epsilon: str
    # Whether to turn on unsat_core for all variables
    unsat_core: bool
    # Whether to simplify output before plotting/saving
    simplify: bool
    # Whether AIMD can additively increase irrespective of losses. If true, the
    # the algorithm is more like cubic and has interesting failure modes
    aimd_incr_irrespective: bool
    # Losses are by default detected using dupacks/timeouts. Oracle emulates
    # losses being detecting using ECN marks signalled to
    # the sender on each loss event.
    loss_oracle: bool = False
    # Whether loss decisions should be deterministic or non-deterministic.
    # Non determinisim is more accurate
    # (considering discrete relaxation of continuous model).
    deterministic_loss: bool = False

    calculate_qbound: bool = False

    # These config variables are calculated automatically
    calculate_qdel: bool = False

    mode_switch: bool = False

    feasible_response: bool = False

    beliefs: bool = False
    beliefs_use_buffer: bool = False
    fix_stale__max_c: bool = False
    fix_stale__min_c: bool = False
    min_maxc_minc_gap_mult: float = 1
    maxc_minc_change_mult: float = 1

    app_limited: bool = False
    app_fixed_avg_rate: bool = False
    app_rate: Optional[float] = None
    app_burst_factor: float = 1

    def __init__(self,
                 N: int,
                 D: int,
                 R: int,
                 T: int,
                 C: float,
                 buf_min: Optional[float],
                 buf_max: Optional[float],
                 dupacks: Optional[float],
                 cca: str,
                 compose: bool,
                 alpha: Optional[float],
                 pacing: bool,
                 epsilon: str,
                 unsat_core: bool,
                 simplify: bool,
                 aimd_incr_irrespective: bool = False,
                 deterministic_loss: bool = False,
                 loss_oracle: bool = False):
        self.__dict__ = locals()
        self.calculate_qdel = cca in ["copa"] or N > 1
        self.calculate_qbound = False

    @staticmethod
    def get_argparse() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-N", "--num-flows", type=int, default=1)
        parser.add_argument("-D", type=int, default=1)
        parser.add_argument("-R", "--rtt", type=int, default=1)
        parser.add_argument("-T", "--time", type=int, default=10)
        parser.add_argument("-C", "--rate", type=float, default=1)
        parser.add_argument("--buf-min", type=float, default=None)
        parser.add_argument("--buf-max", type=float, default=None)
        parser.add_argument("--dupacks", type=float, default=None)
        parser.add_argument(
            "--cca",
            type=str,
            default="const",
            choices=["const", "aimd", "copa", "bbr", "fixed_d", "any"])
        parser.add_argument("--no-compose", action="store_true")
        parser.add_argument("--alpha", type=float, default=None)
        parser.add_argument("--pacing",
                            action="store_const",
                            const=True,
                            default=False)
        parser.add_argument(
            "--epsilon",
            type=str,
            default="zero",
            choices=["zero", "lt_alpha", "lt_half_alpha", "gt_alpha"])
        parser.add_argument("--unsat-core", action="store_true")
        parser.add_argument("--simplify", action="store_true")
        parser.add_argument("--aimd-incr-irrespective", action="store_true")
        parser.add_argument("--deterministic-loss", action="store_true",
                            default=False)
        parser.add_argument("--loss-oracle", action="store_true",
                            default=False)

        return parser

    @classmethod
    def from_argparse(cls, args: argparse.Namespace):
        return cls(args.num_flows, args.D, args.rtt, args.time, args.rate,
                   args.buf_min, args.buf_max, args.dupacks, args.cca,
                   not args.no_compose, args.alpha, args.pacing, args.epsilon,
                   args.unsat_core, args.simplify, args.aimd_incr_irrespective)

    @classmethod
    def default(cls):
        return cls.from_argparse(cls.get_argparse().parse_args(args=[]))
