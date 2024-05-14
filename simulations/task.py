from simulations.monitor import Monitor
from simulations.state import State


class Task:
    """A simple Task. Applications may subclass this
       for holding specific attributes if need be"""

    def __init__(self, id_: str, simulation, is_long_task: bool = False) -> None:
        self.id: str = id_
        self.simulation = simulation
        self.start: int = self.simulation.now
        self.completion_event = self.simulation.event()
        self._is_long_task = is_long_task
        self.state_at_arrival_time: State | None = None

    def set_state(self, state: State) -> None:
        self.state_at_arrival_time = state

    def get_state(self) -> State | None:
        return self.state_at_arrival_time

    def is_long_task(self) -> bool:
        return self._is_long_task

    # Used as a notifier mechanism
    def signal_task_complete(self, piggyback=None):
        if self.completion_event is not None:
            self.completion_event.succeed(piggyback)
