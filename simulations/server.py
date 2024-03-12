from global_sim import Simulation
import simpy
import math
import random
import sys
from monitor import Monitor


class Server():
    """A representation of a physical server that holds resources"""
    def __init__(self, id_, resourceCapacity,
                 serviceTime, serviceTimeModel):
        self.id = id_
        self.serviceTime = serviceTime
        self.serviceTimeModel = serviceTimeModel
        self.queueResource = simpy.Resource(capacity=resourceCapacity, env=Simulation)
        self.serverRRMonitor = Monitor()
        self.waitMon = Monitor()
        self.actMon = Monitor()

    def enqueueTask(self, task):
        executor = Executor(self, task)
        self.serverRRMonitor.observe(1)
        Simulation.process(executor.run())
        # Simulation.activate(executor, executor.run(), Simulation.now)

    def getServiceTime(self):
        serviceTime = 0.0
        if (self.serviceTimeModel == "random.expovariate"):
            serviceTime = random.expovariate(1.0/(self.serviceTime))
        elif (self.serviceTimeModel == "constant"):
            serviceTime = self.serviceTime
        elif(self.serviceTimeModel == "math.sin"):
            serviceTime = self.serviceTime \
                + self.serviceTime \
                * math.sin(1 + Simulation.now/100)
        else:
            print("Unknown service time model")
            sys.exit(-1)

        return serviceTime


class Executor:

    def __init__(self, server, task):
        self.server = server
        self.task = task
        # Simulation.process(self.run(), 'Executor')

    def run(self):
        start = Simulation.now
        queueSizeBefore = len(self.server.queueResource.queue)
        yield Simulation.timeout(0)
        request = self.server.queueResource.request()
        yield request
        waitTime = Simulation.now - start         # W_i
        serviceTime = self.server.getServiceTime()  # Mu_i
        yield Simulation.timeout(serviceTime)
        self.server.queueResource.release(request)

        self.server.waitMon.observe(waitTime)
        self.server.actMon.observe(serviceTime)

        queueSizeAfter = len(self.server.queueResource.queue)
        self.task.sigTaskComplete({"waitingTime": waitTime,
                                   "serviceTime": serviceTime,
                                   "queueSizeBefore": queueSizeBefore,
                                   "queueSizeAfter": queueSizeAfter})
