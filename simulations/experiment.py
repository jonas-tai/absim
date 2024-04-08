from global_sim import Simulation
import server
import client
import workload
import argparse
import random
import constants
import numpy as np
import sys
import muUpdater
from simulations.monitor import Monitor
from simulation_args import SimulationArgs, TimeVaryingArgs, SlowServerArgs
from pathlib import Path
from model_trainer import Trainer
from simulations.plotting import ExperimentPlot


def printMonitorTimeSeriesToFile(fileDesc, prefix, monitor):
    for entry in monitor:
        fileDesc.write("%s %s %s\n" % (prefix, entry[0], entry[1]))


def rlExperimentWrapper(args):
    # Start the models and etc.
    # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    trainer = Trainer(args.num_servers)
    NUM_EPSIODES = 10

    for i_episode in range(NUM_EPSIODES):
        latencies = runExperiment(args, trainer)
    

def runExperiment(args, trainer : Trainer = None):
    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    servers = []
    clients = []
    workloadGens = []

    constants.NW_LATENCY_BASE = args.nwLatencyBase
    constants.NW_LATENCY_MU = args.nwLatencyMu
    constants.NW_LATENCY_SIGMA = args.nwLatencySigma
    constants.NUMBER_OF_CLIENTS = args.numClients

    assert args.expScenario != ""

    serviceRatePerServer = []
    if (args.expScenario == "base"):
        # Start the servers
        for i in range(args.numServers):
            serv = server.Server(i,
                                 resource_capacity=args.serverConcurrency,
                                 service_time=(args.service_time),
                                 service_time_model=args.service_time_model)
            servers.append(serv)
    elif (args.expScenario == "multipleServiceTimeServers"):
        # Start the servers
        for i in range(args.numServers):
            serv = server.Server(i,
                                 resource_capacity=args.serverConcurrency,
                                 service_time=((i + 1) * args.service_time),
                                 service_time_model=args.service_time_model)
            servers.append(serv)
    elif (args.expScenario == "heterogenousStaticServiceTimeScenario"):
        baseServiceTime = args.service_time

        assert args.slowServerFraction >= 0 and args.slowServerFraction < 1.0
        assert args.slowServerSlowness >= 0 and args.slowServerSlowness < 1.0
        assert not (args.slowServerSlowness == 0
                    and args.slowServerFraction != 0)
        assert not (args.slowServerSlowness != 0
                    and args.slowServerFraction == 0)

        if (args.slowServerFraction > 0.0):
            slowServerRate = (args.serverConcurrency *
                              1 / float(baseServiceTime)) * \
                             args.slowServerSlowness
            numSlowServers = int(args.slowServerFraction * args.numServers)
            slowServerRates = [slowServerRate] * numSlowServers

            numFastServers = args.numServers - numSlowServers
            totalRate = (args.serverConcurrency *
                         1 / float(args.service_time) * args.numServers)
            fastServerRate = (totalRate - sum(slowServerRates)) \
                             / float(numFastServers)
            fastServerRates = [fastServerRate] * numFastServers
            serviceRatePerServer = slowServerRates + fastServerRates
        else:
            serviceRatePerServer = [args.serverConcurrency *
                                    1 / float(args.service_time)] * args.numServers

        random.shuffle(serviceRatePerServer)
        # print(sum(serviceRatePerServer), (1/float(baseServiceTime)) * args.numServers)
        assert sum(serviceRatePerServer) > 0.99 * \
               (1 / float(baseServiceTime)) * args.numServers
        assert sum(serviceRatePerServer) <= \
               (1 / float(baseServiceTime)) * args.numServers

        # Start the servers
        for i in range(args.numServers):
            st = 1 / float(serviceRatePerServer[i])
            serv = server.Server(i,
                                 resource_capacity=args.serverConcurrency,
                                 service_time=st,
                                 service_time_model=args.service_time_model)
            servers.append(serv)
    elif (args.expScenario == "timeVaryingServiceTimeServers"):
        assert args.intervalParam != 0.0
        assert args.timeVaryingDrift != 0.0

        # Start the servers
        for i in range(args.numServers):
            serv = server.Server(i,
                                 resource_capacity=args.serverConcurrency,
                                 service_time=(args.service_time),
                                 service_time_model=args.service_time_model)
            mup = muUpdater.MuUpdater(serv, args.intervalParam,
                                      args.service_time,
                                      args.timeVaryingDrift)
            Simulation.process(mup.run())
            servers.append(serv)
    else:
        print("Unknown experiment scenario")
        sys.exit(-1)

    baseDemandWeight = 1.0
    clientWeights = []
    assert args.highDemandFraction >= 0 and args.highDemandFraction < 1.0
    assert args.demandSkew >= 0 and args.demandSkew < 1.0
    assert not (args.demandSkew == 0 and args.highDemandFraction != 0)
    assert not (args.demandSkew != 0 and args.highDemandFraction == 0)

    if (args.highDemandFraction > 0.0 and args.demandSkew >= 0):
        heavyClientWeight = baseDemandWeight * \
                            args.demandSkew / args.highDemandFraction
        numHeavyClients = int(args.highDemandFraction * args.numClients)
        heavyClientWeights = [heavyClientWeight] * numHeavyClients

        lightClientWeight = baseDemandWeight * \
                            (1 - args.demandSkew) / (1 - args.highDemandFraction)
        numLightClients = args.numClients - numHeavyClients
        lightClientWeights = [lightClientWeight] * numLightClients
        clientWeights = heavyClientWeights + lightClientWeights
    else:
        clientWeights = [baseDemandWeight] * args.numClients

    assert sum(clientWeights) > 0.99 * args.numClients
    assert sum(clientWeights) <= args.numClients

    # Start the clients
    for i in range(args.numClients):
        c = client.Client(id_="Client%s" % (i),
                          server_list=servers,
                          replicaSelectionStrategy=args.selectionStrategy,
                          accessPattern=args.accessPattern,
                          replicationFactor=args.replicationFactor,
                          backpressure=args.backpressure,
                          shadowReadRatio=args.shadowReadRatio,
                          rateInterval=args.rateInterval,
                          cubicC=args.cubicC,
                          cubicSmax=args.cubicSmax,
                          cubicBeta=args.cubicBeta,
                          hysterisisFactor=args.hysterisisFactor,
                          demandWeight=clientWeights[i],
                          rate_intervals=args.rate_intervals,
                          trainer=trainer)
        clients.append(c)

    # Start workload generators (analogous to YCSB)
    latencyMonitor = Monitor(name="Latency")

    # This is where we set the inter-arrival times based on
    # the required utilization level and the service time
    # of the overall server pool.
    arrivalRate = 0
    interArrivalTime = 0
    if (len(serviceRatePerServer) > 0):
        print(serviceRatePerServer)
        arrivalRate = (args.utilization * sum(serviceRatePerServer))
        interArrivalTime = 1 / float(arrivalRate)
    else:
        arrivalRate = args.numServers * \
                      (args.utilization * args.serverConcurrency *
                       1 / float(args.service_time))
        interArrivalTime = 1 / float(arrivalRate)

    for i in range(args.numWorkload):
        w = workload.Workload(i, latencyMonitor,
                              clients,
                              args.workloadModel,
                              interArrivalTime * args.numWorkload,
                              args.numRequests / args.numWorkload)
        Simulation.process(w.run())
        workloadGens.append(w)

    # Begin simulation
    Simulation.run(until=args.simulationDuration)

    #
    # print(a bunch of timeseries)
    #

    exp_path = Path('..', args.logFolder, args.expPrefix)

    if not exp_path.exists():
        exp_path.mkdir(parents=True, exist_ok=True)

    pendingRequestsFD = open("../%s/%s_PendingRequests" %
                             (args.logFolder,
                              args.expPrefix), 'w')
    waitMonFD = open("../%s/%s_WaitMon" % (args.logFolder,
                                           args.expPrefix), 'w')
    actMonFD = open("../%s/%s_ActMon" % (args.logFolder,
                                         args.expPrefix), 'w')
    latencyFD = open("../%s/%s_Latency" % (args.logFolder,
                                           args.expPrefix), 'w')
    latencyTrackerFD = open("../%s/%s_LatencyTracker" %
                            (args.logFolder, args.expPrefix), 'w')
    rateFD = open("../%s/%s_Rate" % (args.logFolder,
                                     args.expPrefix), 'w')
    tokenFD = open("../%s/%s_Tokens" % (args.logFolder,
                                        args.expPrefix), 'w')
    receiveRateFD = open("../%s/%s_ReceiveRate" % (args.logFolder,
                                                   args.expPrefix), 'w')
    edScoreFD = open("../%s/%s_EdScore" % (args.logFolder,
                                           args.expPrefix), 'w')
    serverRRFD = open("../%s/%s_serverRR" % (args.logFolder,
                                             args.expPrefix), 'w')

    for clientNode in clients:
        printMonitorTimeSeriesToFile(pendingRequestsFD,
                                     clientNode.id,
                                     clientNode.pendingRequestsMonitor)
        printMonitorTimeSeriesToFile(latencyTrackerFD,
                                     clientNode.id,
                                     clientNode.latencyTrackerMonitor)
        printMonitorTimeSeriesToFile(rateFD,
                                     clientNode.id,
                                     clientNode.rateMonitor)
        printMonitorTimeSeriesToFile(tokenFD,
                                     clientNode.id,
                                     clientNode.tokenMonitor)
        printMonitorTimeSeriesToFile(receiveRateFD,
                                     clientNode.id,
                                     clientNode.receiveRateMonitor)
        printMonitorTimeSeriesToFile(edScoreFD,
                                     clientNode.id,
                                     clientNode.edScoreMonitor)

    if args.print:
        for serv in servers:
            printMonitorTimeSeriesToFile(waitMonFD,
                                         serv.id,
                                         serv.wait_monitor)
            printMonitorTimeSeriesToFile(actMonFD,
                                         serv.id,
                                         serv.act_monitor)
            printMonitorTimeSeriesToFile(serverRRFD,
                                         serv.id,
                                         serv.server_RR_monitor)
            print("------- Server:%s %s ------" % (serv.id, "WaitMon"))
            print("Mean:", serv.wait_monitor.mean())

            print("------- Server:%s %s ------" % (serv.id, "ActMon"))
            print("Mean:", serv.act_monitor.mean())

        print("------- Latency ------")
        print("Mean Latency:", latencyMonitor.mean())
        for p in [50, 95, 99]:
            print(f"p{p} Latency: {latencyMonitor.percentile(p)}")


        printMonitorTimeSeriesToFile(latencyFD, "0",
                                     latencyMonitor)
        assert args.numRequests == len(latencyMonitor)

    plotter = ExperimentPlot()

    plotter.add_data(latencyMonitor)

    return latencyMonitor


# class WorkloadUpdater(Simulation.Process):
#     def __init__(self, workload, value, clients, servers):
#         self.workload = workload
#         self.value = value
#         self.clients = clients
#         self.servers = servers
#         Simulation.Process.__init__(self, name='WorkloadUpdater')

#     def run(self):
#         # while(1):
#             yield Simulation.hold, self,
#             # self.workload.model_param = random.uniform(self.value,
#                                                        # self.value * 40)
#             # old = self.workload.model_param
#             # self.workload.model_param = self.value
#             # self.workload.clientList = self.clients
#             # self.workload.total = \
#             #     sum(client.demandWeight for client in self.clients)
#             # yield Simulation.hold, self, 1000
#             # self.workload.model_param = old
#             self.servers[0].serviceTime = 1000

if __name__ == '__main__':

    args = SimulationArgs()
    # args = TimeVaryingArgs(0.1,5)
    # args = SlowServerArgs(0.5,0.5)
    args.set_policy('expDelay')



    runExperiment(args.args)
