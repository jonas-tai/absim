from dataclasses import dataclass
from typing import List
import torch


@dataclass
class NodeState:
    response_time: int
    outstanding_requests: int
    queue_size: int = 0
    service_time: int = 0
    # Wait time is no feature of ARS, we should experiment with using it
    wait_time: int = 0
    # Probably unnecessary since we want to keep this constant for now, might be interesting to experiment
    # with varying network delays
    twice_network_latency: int = 0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [[self.queue_size, self.service_time, self.wait_time, self.response_time, self.outstanding_requests,
             self.twice_network_latency]])


@dataclass
class State:
    time_since_last_req: int
    # Track the number of requests in the last 1s, 0.5s, 0.1s,...
    request_trend: List[int]
    node_states: List[NodeState]

    def to_tensor(self) -> torch.Tensor:
        node_state_tensor = torch.cat([node_state.to_tensor for node_state in self.node_states], 1)
        general_state = self.request_trend + [self.time_since_last_req]
        general_state_tensor = torch.tensor([general_state])
        return torch.cat((general_state_tensor, node_state_tensor), 1)
