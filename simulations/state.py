from dataclasses import dataclass
from typing import List
import torch
import inspect


@dataclass
class NodeState:
    response_time: float
    outstanding_requests: float
    queue_size: float = 0
    service_time: float = 0
    # Wait time is no feature of ARS, we should experiment with using it
    wait_time: float = 0
    # Probably unnecessary since we want to keep this constant for now, might be interesting to experiment
    # with varying network delays
    twice_network_latency: float = 0
    outstanding_long_requests: int = 0
    outstanding_short_requests: int = 0

    def to_tensor(self, long_requests_ratio: float) -> torch.Tensor:
        if long_requests_ratio > 0:
            return torch.tensor(
                [[self.queue_size, self.service_time, self.wait_time, self.response_time, self.outstanding_requests,
                 self.outstanding_long_requests, self.outstanding_short_requests]],  # self.twice_network_latency
                dtype=torch.float32)
        return torch.tensor(
            [[self.queue_size, self.service_time, self.wait_time, self.response_time, self.outstanding_requests,
              ]], dtype=torch.float32)  # self.twice_network_latency

    @staticmethod
    def get_node_state_size(long_requests_ratio: float) -> int:
        if long_requests_ratio > 0:
            return 7
        return 5


@dataclass
class State:
    time_since_last_req: int
    is_long_request: bool
    # Track the number of requests in the last 1s, 0.5s, 0.1s,...
    request_trend: List[int]
    node_states: List[NodeState]

    def to_tensor(self, long_requests_ratio: float) -> torch.Tensor:
        node_state_tensor = torch.cat(
            [node_state.to_tensor(long_requests_ratio=long_requests_ratio) for node_state in self.node_states], 1)
        general_state = self.request_trend + [self.time_since_last_req,
                                              int(self.is_long_request)] if long_requests_ratio > 0 else self.request_trend + [
            self.time_since_last_req]
        general_state_tensor = torch.tensor([general_state], dtype=torch.float32)
        return torch.cat((general_state_tensor, node_state_tensor), 1)

    @staticmethod
    def get_state_size(num_servers: int, long_requests_ratio: float, num_request_rates: int = 3):
        num_other_features = 2 if long_requests_ratio > 0 else 1
        state_size = num_servers * NodeState.get_node_state_size(
            long_requests_ratio=long_requests_ratio) + num_request_rates + num_other_features
        return state_size
