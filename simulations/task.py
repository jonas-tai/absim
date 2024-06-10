from simulations.monitor import Monitor
from simulations.state import State


class Task:
    """A simple Task. Applications may subclass this
       for holding specific attributes if need be"""

    def __init__(self, id_: str, simulation, is_long_task: bool = False, start: int | None = None, is_duplicate=False, original_id: str | None = None) -> None:
        self.id: str = id_
        self.original_id = id_ if original_id is None else original_id
        self.simulation = simulation
        self.start: int = self.simulation.now if start is None else start
        self.completion_event = self.simulation.event()
        self._is_long_task = is_long_task
        self.state_at_arrival_time: State | None = None
        self.is_duplicate = is_duplicate
        self.has_duplicate = False
        self.q_values = None

    def set_q_values(self, q_values):
        self.q_values = q_values

    def create_duplicate_task(self):
        duplicate_id = f'duplicate_{self.id}'
        duplicate_task = Task(id_=duplicate_id, simulation=self.simulation,
                              is_long_task=self._is_long_task, start=self.start, is_duplicate=True, original_id=self.id)
        if self.state_at_arrival_time is not None:
            duplicate_task.set_state(self.state_at_arrival_time.deep_copy())
        self.has_duplicate = True
        return duplicate_task

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
