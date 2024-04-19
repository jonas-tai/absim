class MuUpdater:

    def __init__(self, server, intervalParam, serviceTime, rateChangeFactor, simulation):
        self.server = server
        self.intervalParam = intervalParam
        self.serviceTime = serviceTime
        self.rateChangeFactor = rateChangeFactor
        self.simulation = simulation
        self.simulation.process(self.run())
        self.is_slow = True

    def run(self):
        while (1):
            yield self.simulation.timeout(0)

            if (self.simulation.random.uniform(0, 1.0) >= 0.5):
                rate = 1 / float(self.serviceTime)
                self.server.service_time = 1 / float(rate)
            else:
                if self.is_slow:
                    rate = 1 / float(self.serviceTime)
                    rate += self.rateChangeFactor * rate
                    self.server.service_time = 1 / float(rate)
                self.is_slow = False
            # print('Test')
            # print(self.rateChangeFactor)
            # print(self.intervalParam)
            yield self.simulation.timeout(self.intervalParam)
