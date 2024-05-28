class MuUpdater:

    def __init__(self, server, interval_param, service_time, rate_change_factor, simulation):
        self.server = server
        self.interval_param = interval_param
        self.service_time = service_time
        self.rate_change_factor = rate_change_factor
        self.simulation = simulation
        self.simulation.process(self.run())
        self.is_fast = True

    def run(self):
        while (1):
            yield self.simulation.timeout(0)

            if (self.simulation.random.uniform(0, 1.0) >= 0.5):
                # rate = 1 / float(self.service_time)
                # self.server.service_time = 1 / float(rate)
                self.server.SERVICE_TIME_FACTOR = 1
            else:
                if self.is_fast:
                    self.server.SERVICE_TIME_FACTOR = self.rate_change_factor
                    # rate = 1 / float(self.service_time)
                    # rate += self.rate_change_factor * rate
                    # self.server.service_time = 1 / float(rate)
                self.is_fast = False
            # print('Test')
            # print(self.rateChangeFactor)
            # print(self.intervalParam)
            yield self.simulation.timeout(self.interval_param)
