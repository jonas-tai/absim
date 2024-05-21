from dataclasses import dataclass
from typing import List
import torch
from sklearn.preprocessing import PolynomialFeatures


@dataclass
class NodeState:
    response_time: float
    outstanding_requests: float
    ars_score: float
    queue_size: float = 0
    service_time: float = 0
    # Wait time is no feature of ARS, we should experiment with using it
    wait_time: float = 0
    # Probably unnecessary since we want to keep this constant for now, might be interesting to experiment
    # with varying network delays
    twice_network_latency: float = 0
    outstanding_long_requests: int = 0
    outstanding_short_requests: int = 0


@dataclass
class State:
    time_since_last_req: int
    is_long_request: bool
    # Track the number of requests in the last 1s, 0.5s, 0.1s,...
    request_trend: List[int]
    node_states: List[NodeState]


class StateParser:
    def __init__(self, num_servers: int, long_requests_ratio: float, num_request_rates: int, poly_feat_degree: int) -> None:
        self.num_servers = num_servers
        self.long_requests_ratio = long_requests_ratio
        self.num_request_rates = num_request_rates
        self.poly_feat_degree = poly_feat_degree

    def create_dummy_state(self) -> State:
        node_states = [NodeState(response_time=0.0, outstanding_requests=0.0, ars_score=0.0)
                       for _ in range(self.num_servers)]
        dummy_state = State(time_since_last_req=0, is_long_request=False, request_trend=[
                            0 for _ in range(self.num_request_rates)], node_states=node_states)
        return dummy_state

    # def get_node_state_size(self) -> int:
    #     if self.long_requests_ratio > 0:
    #         return 7
    #     return 5

    def get_state_size(self):
        # num_other_features = 2 if self.long_requests_ratio > 0 else 1
        # state_size = self.num_servers * self.get_node_state_size(
        #     long_requests_ratio=self.long_requests_ratio) + self.num_request_rates + num_other_features

        dummy_state = self.create_dummy_state()
        dummy_state_tensor = self.state_to_tensor(dummy_state)
        state_length = dummy_state_tensor.size(dim=1)
        return state_length  # 1540

    def node_state_to_tensor(self, node_state: NodeState) -> torch.Tensor:
        state_features = [node_state.queue_size, node_state.service_time,
                          node_state.wait_time, node_state.response_time, node_state.outstanding_requests]  # node_state.ars_score
        # state_features = [node_state.ars_score]
        # node_state.twice_network_latency

        if self.long_requests_ratio > 0:
            state_features += [node_state.outstanding_long_requests, node_state.outstanding_short_requests]

        return torch.tensor([state_features], dtype=torch.float32)

    def state_to_tensor(self, state: State) -> torch.Tensor:
        node_state_tensor = torch.cat(
            [self.node_state_to_tensor(node_state) for node_state in state.node_states], 1)

        general_state = state.request_trend + [state.time_since_last_req]
        # general_state = []
        if self.long_requests_ratio > 0:
            general_state += [int(state.is_long_request)]
        general_state_tensor = torch.tensor([general_state], dtype=torch.float32)

        state_tensor = torch.cat((general_state_tensor, node_state_tensor), 1)

        # Add polynomial and interaction features
        poly = PolynomialFeatures(self.poly_feat_degree)
        poly_state = poly.fit_transform(state_tensor)
        return torch.tensor(poly_state, dtype=torch.float32)
