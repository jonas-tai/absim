from global_sim import Simulation
import random


class MuUpdater:

    def __init__(self, server, intervalParam, serviceTime, rateChangeFactor):
        self.server = server
        self.intervalParam = intervalParam
        self.serviceTime = serviceTime
        self.rateChangeFactor = rateChangeFactor
        Simulation.process(self.run(), 'MuUpdater')

    def run(self):
        while (1):
            yield Simulation.timeout(0)

            if (random.uniform(0, 1.0) >= 0.5):
                rate = 1 / float(self.serviceTime)
                self.server.serviceTime = 1 / float(rate)
            else:
                rate = 1 / float(self.serviceTime)
                rate += self.rateChangeFactor * rate
                self.server.serviceTime = 1 / float(rate)
            # print(Simulation.now, self.server.id, self.server.serviceTime)
            yield Simulation.hold, self, self.intervalParam
