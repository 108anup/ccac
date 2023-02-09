import z3
import numpy as np
from typing import Any, List, Optional, Tuple, Union

import pyz3_utils
from pyz3_utils import MySolver

from .config import ModelConfig


class Variables(pyz3_utils.Variables):
    ''' Some variables that everybody uses '''

    def __init__(self, c: ModelConfig, s: MySolver,
                 name: Optional[str] = None):
        T = c.T

        # Add a prefix to all names so we can have multiple Variables instances
        # in one solver
        if name is None:
            pre = ""
        else:
            pre = name + "__"
        self.pre = pre

        # Naming convention: X_f denotes per-flow values (note, we only study
        # the single-flow case in the paper)

        # Cumulative number of bytes sent by flow n till time t
        self.A_f = np.array([[
            s.Real(f"{pre}arrival_{n},{t}") for t in range(T)]
            for n in range(c.N)])
        # Sum of A_f across all flows
        self.A = np.array([s.Real(f"{pre}tot_arrival_{t}") for t in range(T)])
        # Congestion window for flow n at time t
        self.c_f = np.array([[
            s.Real(f"{pre}cwnd_{n},{t}") for t in range(T)]
            for n in range(c.N)])
        # Pacing rate for flow n at time t
        self.r_f = np.array([[
            s.Real(f"{pre}rate_{n},{t}") for t in range(T)]
            for n in range(c.N)])
        # Cumulative number of losses detected (by duplicate acknowledgements
        # or timeout) by flow n till time t
        self.Ld_f = np.array([[
            s.Real(f"{pre}loss_detected_{n},{t}")
            for t in range(T)]
            for n in range(c.N)])
        # Cumulative number of bytes served from the server for flow n till
        # time t. These acks corresponding to these bytes will reach the sender
        # at time t+c.R
        self.S_f = np.array([[
            s.Real(f"{pre}service_{n},{t}") for t in range(T)]
            for n in range(c.N)])
        if(c.feasible_response):
            assert(c.N == 1)
            self.S_choice = np.array(
                [s.Real(f"{pre}tot_service_choice_{t}") for t in range(T)])
        # Sum of S_f across all flows
        self.S = np.array([s.Real(f"{pre}tot_service_{t}") for t in range(T)])
        # Cumulative number of bytes lost for flow n till time t
        self.L_f = np.array([[
            s.Real(f"{pre}losts_{n},{t}") for t in range(T)]
            for n in range(c.N)])
        # Sum of L_f for all flows
        self.L = np.array([s.Real(f"{pre}tot_lost_{t}") for t in range(T)])
        # Cumulative number of bytes wasted by the server till time t
        self.W = np.array([s.Real(f"{pre}wasted_{t}") for t in range(T)])
        # Whether or not flow n is timing out at time t
        if(not c.loss_oracle):
            self.timeout_f = np.array([[
                s.Bool(f"{pre}timeout_{n},{t}") for t in range(T)]
                for n in range(c.N)])

        # If qdel[t][dt] is true, it means that the bytes exiting at t were
        # input at time t - dt. If out[t] == out[t-1], then qdel[t][dt] ==
        # qdel[t-1][dt], since qdel isn't really defined (since no packets were
        # output), so we default to saying we experience the RTT of the last
        # received packet.

        # This is only computed when calculate_qdel=True since not all CCAs
        # require it. Of the CCAs implemented so far, only Copa requires it
        if c.calculate_qdel:
            self.qdel = np.array([[
                s.Bool(f"{pre}qdel_{t},{dt}") for dt in range(T)]
                for t in range(T)])

        # qbound[t][dt] is true iff the queuing delay experienced by bytes
        # recieved at time t is greater than or equal to dt. In other words,
        # qbound[t][dt] is true iff S[t] <= A[t-dt] - L[t-dt]. If S[t] ==
        # S[t-1], then no new bytes were serviced, so qbound is undefined, we
        # copy the qbound of last serviced byte. If dt = 0 then qbound is by
        # definition True as queing delay >= 0. Queing delay at time t can be
        # greater than t, if S[t] < A[0] - L[0], then queueing delay is greater
        # than t. When dt > t, we don't know when the byte was sent
        if(c.calculate_qbound):
            self.qbound = np.array([[
                s.Bool(f"{pre}qbound_{t},{dt}")
                for dt in range(T)]
                for t in range(T)])

            # Used by CCA to ensure it reacts at most once per congestion event
            self.exceed_queue_f = [[s.Bool(f"{pre}exceed_queue_{n},{t}")
                                    for t in range(c.T)]
                                   for n in range(c.N)]
            self.last_decrease_f = [[s.Real(f"{pre}last_decrease_{n},{t}")
                                     for t in range(c.T)]
                                    for n in range(c.N)]
            # Cegis generator var
            # This is in multiples of Rm
            self.qsize_thresh = s.Real(f"{pre}Gen__const_qsize_thresh")
            assert isinstance(self.qsize_thresh, z3.ArithRef)

        if(c.mode_switch):
            self.mode_f = np.array([[
                s.Bool(f"{pre}Def__mode_{n},{t}") for t in range(c.T)]
                for n in range(c.N)])

        # This is for the non-composing model where waste is allowed only when
        # A - L and S come within epsilon of each other. See in 'config' for
        # how epsilon can be configured
        if not c.compose:
            self.epsilon = s.Real(f"{pre}epsilon")

        # The number of dupacks that need to arrive before we declare that a
        # loss has occured by dupacks. Z3 can usually pick any amount. You can
        # also set dupacks = 3 * alpha to emulate the usual behavior
        if(not c.loss_oracle):
            if c.dupacks is None:
                self.dupacks = s.Real(f"{pre}dupacks")
                s.add(self.dupacks >= 0)
            else:
                self.dupacks = c.dupacks

        # The MSS. Since C=1 (arbitrary units), C / alpha sets the link rate in
        # MSS/timestep. Typically we allow Z3 to pick any value it wants to
        # search through the set of all possible link rates
        if c.alpha is None:
            self.alpha = s.Real(f"{pre}alpha")
            s.add(self.alpha > 0)
        else:
            self.alpha = c.alpha

        self.C0 = s.Real(f"{pre}initial_tokens")

        if(c.beliefs):
            self.max_c = np.array([[
                s.Real(f"{pre}max_c_{n},{t}")
                for t in range(T)]
                for n in range(c.N)])
            self.min_c = np.array([[
                s.Real(f"{pre}min_c_{n},{t}")
                for t in range(T)]
                for n in range(c.N)])
            self.max_qdel = np.array([[
                s.Real(f"{pre}max_qdel_{n},{t}")
                for t in range(T)]
                for n in range(c.N)])
            self.min_qdel = np.array([[
                s.Real(f"{pre}min_qdel_{n},{t}")
                for t in range(T)]
                for n in range(c.N)])

            if(c.buf_min is not None and c.beliefs_use_buffer):
                self.max_buffer = np.array([[
                    s.Real(f"{pre}max_buffer_{n},{t}")
                    for t in range(T)]
                    for n in range(c.N)])
                self.min_buffer = np.array([[
                    s.Real(f"{pre}min_buffer_{n},{t}")
                    for t in range(T)]
                    for n in range(c.N)])
            self.start_state_f = np.array([
                s.Int(f"{pre}belief_state_state_{n}") for n in range(c.N)])
            for n in range(c.N):
                s.add(z3.And(self.start_state_f[n] >= 0, self.start_state_f[n] < c.T-1))

        if (c.app_limited):
            self.app_limits = np.array([[
                s.Real(f"{pre}app_limits_{n},{t}")
                for t in range(c.T)]
                for n in range(c.N)])

            if(c.app_fixed_avg_rate):
                self.app_rate = z3.Real(f"{pre}app_rate")

                if (c.beliefs):
                    self.max_app_rate = np.array([[
                        s.Real(f"{pre}max_app_rate_{n},{t}")
                        for t in range(T)]
                        for n in range(c.N)])
                    self.min_app_rate = np.array([[
                        s.Real(f"{pre}min_app_rate_{n},{t}")
                        for t in range(T)]
                        for n in range(c.N)])


class VariableNames:
    ''' Class with the same structure as Variables, but with just the names '''

    def __init__(self, v: Variables):
        for x in v.__dict__:
            if (isinstance(v.__dict__[x], list)
                    or isinstance(v.__dict__[x], np.ndarray)):
                self.__dict__[x] = self.to_names(v.__dict__[x])
            else:
                self.__dict__[x] = str(v.__dict__[x])

    @classmethod
    def to_names(cls, x: Union[List[Any], np.ndarray]):
        res = []
        for y in x:
            if isinstance(y, list) or isinstance(y, np.ndarray):
                res.append(cls.to_names(y))
            else:
                if type(y) in [bool, int, float, tuple]:
                    res.append(y)
                else:
                    res.append(str(y))
        return np.array(res)
