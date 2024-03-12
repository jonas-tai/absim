from global_sim import Simulation


class Task():
    """A simple Task. Applications may subclass this
       for holding specific attributes if need be"""
    def __init__(self, id_, latencyMonitor):
        self.id = id_
        self.start = Simulation.now
        self.completionEvent = Simulation.event()
        self.latencyMonitor = latencyMonitor

    # Used as a notifier mechanism
    def sigTaskComplete(self, piggyback=None):
        if (self.completionEvent is not None):
            self.completionEvent.succeed(piggyback)
