class MuUpdater:

    def __init__(self, server, intervalParam, serviceTime, rateChangeFactor, simulation):
        self.server = server
        self.intervalParam = intervalParam
        self.serviceTime = serviceTime
        self.rateChangeFactor = rateChangeFactor
        self.simulation = simulation
        self.simulation.process(self.run())

    def run(self):
        while (1):
            yield self.simulation.timeout(0)

            if (self.simulation.random.uniform(0, 1.0) >= 0.5):
                rate = 1 / float(self.serviceTime)
                self.server.service_time = 1 / float(rate)
            else:
                rate = 1 / float(self.serviceTime)
                rate += self.rateChangeFactor * rate
                self.server.service_time = 1 / float(rate)
            # print(self.simulation.now, self.server.id, self.server.serviceTime)
            yield self.simulation.timeout(self.intervalParam)
