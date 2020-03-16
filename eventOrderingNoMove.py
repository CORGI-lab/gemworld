import networkx as nx
import matplotlib.pyplot as plt

pullWords = ['pulling', 'yanking']
crossWords = ['crossing', 'going over']
movementWords = ['moving', 'heading', 'advancing', 'running', 'walking', 'travelling', 'going']
obtainmentWords = ['getting', 'obtaining', 'taking', 'grabbing', 'picking up', 'procuring', 'fetching']
unlockWords = ['unlocking', 'opening', 'unlatching']
reachWords = ['reaching', 'getting to', 'arriving at', 'entering', 'landing on']
exitWords = ['exiting', 'escaping', 'leaving', 'getting out']
pullFWords = ['pull', 'yank']
crossFWords = ['cross', 'go over']
moveFWords = ['move', 'head', 'advance', 'run', 'walk', 'travel', 'go']
obtainFWords = ['get', 'obtain', 'take', 'grab', 'pick up', 'procure', 'fetch']
unlockFWords = ['unlock', 'open', 'unlatch']
reachFWords = ['reach', 'get to', 'arrive at', 'enter', 'land on']
exitFWords = ['exit', 'escape', 'leave', 'get out']
safeWords = ['unharmed', 'unhurt', 'safely', 'in one piece']
flavorWords = ['with gusto.', 'slowly but surely.', 'in style.']

# exploreEvent = 0
# deadEvent = 1
# exitEvent = 2
# reachSafeEvent = 3
# door1Event = 4
# door2Event = 5
# key1Event = 6
# key2Event = 7
# gem1Event = 8
# gem2Event = 9
# moveSafeEvent = 10
# moveDoor1Event = 11
# moveDoor2Event = 12
# moveKey1Event = 13
# moveKey2Event = 14
# moveGem1Event = 15
# moveGem2Event = 16
# moveBridgeEvent = 17
# moveLeverEvent = 18
# crossBridgeEvent = 19
# liftBridgeEvent = 20
# lowerBridgeEvent = 21

exploreEvent = "explore"
deadEvent = "dead"
exitEvent = "exit"
# reachSafeEvent = "safe"
door1Event = "open door 1"
door2Event = "open door 2"
key1Event = "key 1"
key2Event = "key 2"
gem1Event = "gem 1"
gem2Event = "gem 2"
# moveSafeEvent = "move to safe"
# moveDoor1Event = "move to door 1"
# moveDoor2Event = "move to door 2"
# moveKey1Event = "move to key 1"
# moveKey2Event = "move to key 2"
# moveGem1Event = "move to gem 1"
# moveGem2Event = "move to gem 1"
moveBridgeEvent = "move to bridge"
moveLeverEvent = "move to lever "
crossBridgeEvent = "cross bridge"
liftBridgeEvent = "lift bridge"
lowerBridgeEvent = "lower bridge"


# if there are multiple of the same events, which event probability do I keep?
# highest, lowest, average
def ground(beams):
    events = set()
    for i in range(len(beams)):
        found = False
        if not found:
            words = beams[i].split()
            pull = False
            cross = False
            movement = False
            obtainment = False
            unlock = False
            reach = False
            exit = False
            pullF = False
            crossF = False
            moveF = False
            obtainF = False
            unlockF = False
            reachF = False
            exitF = False

            for i in range(len(words)):
                if words[i] in pullWords:
                    pull = True
                    break
                elif words[i] in crossWords:
                    cross = True
                    break
                elif words[i] in movementWords:
                    movement = True
                    break
                elif words[i] in obtainmentWords:
                    obtainment = True
                    break
                elif words[i] in unlockWords:
                    unlock = True
                    break
                elif words[i] in reachWords:
                    reach = True
                    break
                elif words[i] in exitWords:
                    exit = True
                    break
                elif words[i] in pullFWords:
                    pullF = True
                    break
                elif words[i] in crossFWords:
                    crossF = True
                    break
                elif words[i] in moveFWords:
                    moveF = True
                    break
                elif words[i] in obtainFWords:
                    obtainF = True
                    break
                elif words[i] in unlockFWords:
                    unlockF = True
                    break
                elif words[i] in reachFWords:
                    reachF = True
                    break
                elif words[i] in exitFWords:
                    exitF = True
                    break
                if i + 1 < len(words) - 1:
                    if words[i] + " " + words[i + 1] in pullWords:
                        pull = True
                        break
                    elif words[i] + " " + words[i + 1] in crossWords:
                        cross = True
                        break
                    elif words[i] + " " + words[i + 1] in movementWords:
                        movement = True
                        break
                    elif words[i] + " " + words[i + 1] in obtainmentWords:
                        obtainment = True
                        break
                    elif words[i] + " " + words[i + 1] in unlockWords:
                        unlock = True
                        break
                    elif words[i] + " " + words[i + 1] in reachWords:
                        reach = True
                        break
                    elif words[i] + " " + words[i + 1] in exitWords:
                        exit = True
                        break
                    elif words[i] + " " + words[i + 1] in pullFWords:
                        pullF = True
                        break
                    elif words[i] + " " + words[i + 1] in crossFWords:
                        crossF = True
                        break
                    elif words[i] + " " + words[i + 1] in moveFWords:
                        moveF = True
                        break
                    elif words[i] + " " + words[i + 1] in obtainFWords:
                        obtainF = True
                        break
                    elif words[i] + " " + words[i + 1] in unlockFWords:
                        unlockF = True
                        break
                    elif words[i] + " " + words[i + 1] in reachFWords:
                        reachF = True
                        break
                    elif words[i] + " " + words[i + 1] in exitFWords:
                        exitF = True
                        obtainF = False
                        break

            if (pull or pullF):
                if 'lever' in words:
                    if 'lift' in words or 'lifting' in words:
                        found = True
                        events.add(liftBridgeEvent)
                        # events.add((liftBridgeEvent, beams[1][i]))
                    elif 'lower' in words or 'lowering' in words:
                        found = True
                        events.add(lowerBridgeEvent)
                        # events.add((lowerBridgeEvent, beams[1][i]))

            if (cross or crossF):
                if 'bridge' in words:
                    found = True
                    events.add(crossBridgeEvent)
                    # events.add((crossBridgeEvent, beams[1][i]))

            elif (movement or moveF):
                if 'lever' in words:
                    found = True
                    events.add(moveLeverEvent)
                    # events.add((moveLeverEvent, beams[1][i]))
                elif 'bridge' in words:
                    found = True
                    events.add(crossBridgeEvent)
                    # events.add((crossBridgeEvent, beams[1][i]))
                elif 'gem 1' in beams[i]:
                    found = True
                    events.add(gem1Event)
                    # events.add((gem1Event, beams[1][i]))
                elif 'gem 2' in beams[i]:
                    found = True
                    events.add(gem2Event)
                    # events.add((gem2Event, beams[1][i]))
                elif 'door 1' in beams[i]:
                    found = True
                    events.add(door1Event)
                    # events.add((door1Event, beams[1][i]))
                elif 'door 2' in beams[i]:
                    found = True
                    events.add(door2Event)
                    # events.add((door2Event, beams[1][i]))
                elif 'key 1' in beams[i]:
                    found = True
                    events.add(key1Event)
                    # events.add((key1Event, beams[1][i]))
                elif 'key 2' in beams[i]:
                    found = True
                    events.add(key2Event)
                    # events.add((key2Event, beams[1][i]))
                elif 'safe spot' in beams[i]:
                    found = True
                    events.add(exitEvent)
                    # events.add((exitEvent, beams[1][i]))


            elif (obtainment or obtainF):
                if 'gem 1' in beams[i]:
                    found = True
                    events.add(gem1Event)
                    # events.add((gem1Event, beams[1][i]))
                elif 'gem 2' in beams[i]:
                    found = True
                    events.add(gem2Event)
                    # events.add((gem2Event, beams[1][i]))
                elif 'key 1' in beams[i]:
                    found = True
                    events.add(key1Event)
                    # events.add((key1Event, beams[1][i]))
                elif 'key 2' in beams[i]:
                    found = True
                    events.add(key2Event)
                    # events.add((key2Event, beams[1][i]))

            elif (unlock or unlockF):
                if 'door 1' in beams[i]:
                    found = True
                    events.add(door1Event)
                    # events.add((door1Event, beams[1][i]))

                elif 'door 2' in beams[i]:
                    found = True
                    events.add(door2Event)
                    # events.add((door2Event, beams[1][i]))

            elif reach or reachF:
                if 'safe spot' in beams[i]:
                    found = True
                    events.add(exitEvent)
                    # events.add((exitEvent, beams[1][i]))

            elif exit or exitF:
                found = True
                events.add(exitEvent)
                # events.add((exitEvent, beams[1][i]))

            # elif 'died' in words:
            #     found = True
            #     events.add((deadEvent, beams[1][i]))
            #
            # elif 'exploring' in words or 'explore' in words:
            #     found = True
            #     events.add((exploreEvent, beams[1][i]))
    return events


# start from beams of last move
# ground beam sentences to an event and create node for each unique event
# keep going backwards through moves
# for each move
# ground its beams to events
# create node for each event not in the graph yet
# draw edge going from newly created node to all other nodes in the graph

def main():
    G = nx.DiGraph()
    text_file = open("/Users/adriennecorwin/Research/Traces/00/optimalTraces.txt", "r")
    steps = text_file.readlines()
    stepBeams = []  # list of tuples ([beam strings, beam probabilities)]
    for step in steps:
        info = step.split(';')
        beams = info[3:13]
        # beamStr = []
        # for beam in beams:
        #     # b = beam.split(',')
        #     beamStr.append(beam)
        stepBeams.append(beams)

    stepCount = len(stepBeams) - 1

    groundedEvents = ground(stepBeams[stepCount])

    # eventsToDelete = []
    # for event in groundedEvents:
    #     if event[0] == exploreEvent:
    #         eventsToDelete.append(event)
    #     if event[0] == deadEvent:
    #         eventsToDelete.append(event)
    # for event in eventsToDelete:
    #     groundedEvents.remove(event)

    if len(groundedEvents) > 0:
        for event in groundedEvents:
            G.add_node(event)

    stepCount -= 1
    while stepCount >= 0:
        groundedEvents = ground(stepBeams[stepCount])
        eventsToDelete = []
        # for event in groundedEvents:
        #     if event[0] == exploreEvent:
        #         eventsToDelete.append(event)
        #     if event[0] == deadEvent:
        #         eventsToDelete.append(event)
        # for event in eventsToDelete:
        #     groundedEvents.remove(event)
        if len(groundedEvents) > 0:
            nodes = list(G.nodes)
            for event in groundedEvents:
                if event not in nodes:
                    G.add_node(event)
                    for node in nodes:
                        G.add_edge(event, node)  # should weight of edges all be the same

        stepCount -= 1
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


if __name__ == "__main__": main()
