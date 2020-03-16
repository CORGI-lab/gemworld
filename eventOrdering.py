import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

# import seq2seq.seq2seqbeamsearch


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

exploreEvent = "explore"
deadEvent = "dead"
exitEvent = "exit"
reachSafeEvent = "safe"
door1Event = "open door 1"
door2Event = "open door 2"
key1Event = "key 1"
key2Event = "key 2"
gem1Event = "gem 1"
gem2Event = "gem 2"
moveSafeEvent = "move safe"
moveDoor1Event = "move door 1"
moveDoor2Event = "move door 2"
moveKey1Event = "move key 1"
moveKey2Event = "move key 2"
moveGem1Event = "move gem 1"
moveGem2Event = "move gem 2"
moveBridgeEvent = "move bridge"
moveLeverEvent = "move lever"
crossBridgeEvent = "cross bridge"
liftBridgeEvent = "lift bridge"
lowerBridgeEvent = "lower bridge"


#if there are multiple of the same events, which event probability do I keep?
#highest, lowest, average
def ground(beams): 
    # events = set()
    events = []
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

            for j in range(len(words)):
                if words[j] in pullWords:
                    pull = True
                    break
                elif words[j] in crossWords:
                    cross = True
                    break
                elif words[j] in movementWords:
                    movement = True
                    break
                elif words[j] in obtainmentWords:
                    obtainment = True
                    break
                elif words[j] in unlockWords:
                    unlock = True
                    break
                elif words[j] in reachWords:
                    reach = True
                    break
                elif words[j] in exitWords:
                    exit = True
                    break
                elif words[j] in pullFWords:
                    pullF = True
                    break
                elif words[j] in crossFWords:
                    crossF = True
                    break
                elif words[j] in moveFWords:
                    moveF = True
                    break
                elif words[j] in obtainFWords:
                    obtainF = True
                    break
                elif words[j] in unlockFWords:
                    unlockF = True
                    break
                elif words[j] in reachFWords:
                    reachF = True
                    break
                elif words[j] in exitFWords:
                    exitF = True
                    break
                if j + 1 < len(words) - 1:
                    if words[j] + " " + words[j + 1] in pullWords:
                        pull = True
                        break
                    elif words[j] + " " + words[j + 1] in crossWords:
                        cross = True
                        break
                    elif words[j] + " " + words[j + 1] in movementWords:
                        movement = True
                        break
                    elif words[j] + " " + words[j + 1] in obtainmentWords:
                        obtainment = True
                        break
                    elif words[j] + " " + words[j + 1] in unlockWords:
                        unlock = True
                        break
                    elif words[j] + " " + words[j + 1] in reachWords:
                        reach = True
                        break
                    elif words[j] + " " + words[j + 1] in exitWords:
                        exit = True
                        break
                    elif words[j] + " " + words[j + 1] in pullFWords:
                        pullF = True
                        break
                    elif words[j] + " " + words[j + 1] in crossFWords:
                        crossF = True
                        break
                    elif words[j] + " " + words[j + 1] in moveFWords:
                        moveF = True
                        break
                    elif words[j] + " " + words[j + 1] in obtainFWords:
                        obtainF = True
                        break
                    elif words[j] + " " + words[j + 1] in unlockFWords:
                        unlockF = True
                        break
                    elif words[j] + " " + words[j + 1] in reachFWords:
                        reachF = True
                        break
                    elif words[j] + " " + words[j + 1] in exitFWords:
                        exitF = True
                        obtainF = False
                        break

            if (pull or pullF):
                if 'lever' in words:
                    if 'lift' in words or 'lifting' in words:
                        found = True
                        # events.add(liftBridgeEvent)
                        events.append(liftBridgeEvent)
                    elif 'lower' in words or 'lowering' in words:
                        found = True
                        # events.add(lowerBridgeEvent)
                        events.append(lowerBridgeEvent)
            if (cross or crossF):
                if 'bridge' in words:
                    found = True
                    # events.add(crossBridgeEvent)
                    events.append(crossBridgeEvent)
            elif (movement or  moveF):
                if 'lever' in words:
                    found = True
                    # events.add(moveLeverEvent)
                    # events.append(moveLeverEvent)
                elif 'bridge' in words:
                    found = True
                    # events.add(moveBridgeEvent)
                    # events.append(moveBridgeEvent)
                elif 'gem 1' in beams[i]:
                    found = True
                    # events.add(moveGem1Event)
                    # events.append(moveGem1Event)
                elif 'gem 2' in beams[i]:
                    found = True
                    # events.add(moveGem2Event)
                    # events.append(moveGem2Event)
                elif 'door 1' in beams[i]:
                    found = True
                    # events.add(moveDoor1Event)
                    # events.append(moveDoor1Event)
                elif 'door 2' in beams[i]:
                    found = True
                    # events.add(moveDoor2Event)
                    # events.append(moveDoor2Event)
                elif 'key 1' in beams[i]:
                    found = True
                    # events.add(moveKey1Event)
                    # events.append(moveKey1Event)
                elif 'key 2' in beams[i]:
                    found = True
                    # events.add(moveKey2Event)
                    # events.append(moveKey2Event)
                elif 'safe spot' in beams[i]:
                    found = True
                    # events.add(moveSafeEvent)
                    # events.append(moveSafeEvent)

            elif (obtainment or obtainF):
                if 'gem 1' in beams[i]:
                    found = True
                    # events.add(gem1Event)
                    events.append(gem1Event)
                elif 'gem 2' in beams[i]:
                    found = True
                    # events.add(gem2Event)
                    events.append(gem2Event)
                elif 'key 1' in beams[i]:
                    found = True
                    # events.add(key1Event)
                    events.append(key1Event)
                elif 'key 2' in beams[i]:
                    found = True
                    # events.add(key2Event)
                    events.append(key2Event)

            elif (unlock or unlockF):
                if 'door 1' in beams[i]:
                    found = True
                    # events.add(door1Event)
                    events.append(door1Event)

                elif 'door 2' in beams[i]:
                    found = True
                    # events.add(door2Event)
                    events.append(door2Event)


            elif reach or reachF:
                if 'safe spot' in beams[i]:
                    found = True
                    # events.add(exitEvent)
                    events.append(exitEvent)

            elif exit or exitF:
                found = True
                # events.add(exitEvent)
                events.append(exitEvent)

            # elif 'died' in words:
            #     found = True
            #     events.add((deadEvent, beams[1][i]))
            #
            # elif 'exploring' in words or 'explore' in words:
            #     found = True
            #     events.add((exploreEvent, beams[1][i]))
    return events


#start from beams of last move
#ground beam sentences to an event and create node for each unique event
#keep going backwards through moves
#for each move
#ground its beams to events
#create node for each event not in the graph yet
#draw edge going from newly created node to all other nodes in the graph

startingPos = ["00", "32", "06", "37", "90", "83", "98", "75", "53", "46"]
graphs = {}

# def main():
# hidden_size = 256
# encoder = seq2seq.seq2seqbeamsearch.EncoderRNN(seq2seq.seq2seqbeamsearch.inputState.n_words, hidden_size).to(
#     seq2seq.seq2seqbeamsearch.device)
# decoder = seq2seq.seq2seqbeamsearch.AttnDecoderRNN(hidden_size, seq2seq.seq2seqbeamsearch.outputState.n_words,
#                                                    dropout_p=0.1).to(seq2seq.seq2seqbeamsearch.device)
# model = seq2seq.seq2seqbeamsearch.torch.load("seq2seq/modelOnlyFuture500000.tar")
# encoder.load_state_dict(model['en'])
# decoder.load_state_dict(model['de'])

G = nx.MultiDiGraph()

for pos in startingPos:
# text_file = open("/Users/adriennecorwin/Research/Traces/" + "00" + "/optimalTracesOnlyCurrent.txt", "r")
# currentToFutureDict = {}
# currentStates = text_file.readlines()
# for currentState in currentStates:
#     output_words, attentions, beams = seq2seq.seq2seqbeamsearch.evaluate(encoder, decoder, currentState.strip("\n"))
#     currentState = currentState.split(".")[0][38:]
#     currentToFutureDict[currentState] = beams
#
# with open("/Users/adriennecorwin/Research/Traces/" + "00" + "/optimalTracesGraphDict.txt ", "w") as f:
#     for key in currentToFutureDict.keys():
#         line = key + "\t"
#         for sentence in currentToFutureDict[key]:
#             line += sentence + " "
#         line += "\n"
#         f.write(line)

    with open("/Users/adriennecorwin/Research/Traces/" + pos + "/optimalTracesGraphDict.txt ", "r") as f:
        for line in f:
            line = line.split("\t")


            currentGrounded = list(ground([line[0]]))
            futureGrounded = list(ground(line[1].strip("\n").split(".")[:10]))

            if len(currentGrounded) > 0 and len(futureGrounded) > 0:
                nodes = list(G.nodes)
                if currentGrounded[0] not in nodes:
                    G.add_node(currentGrounded[0])

                for event in futureGrounded:
                    if event not in nodes:
                        G.add_node(event)
                    G.add_edge(currentGrounded[0], event)

nodes = list(G.nodes)
numEdges = {}
for node1 in nodes:
    for node2 in nodes:
        if node1 != node2:
            if G.number_of_edges(node1, node2) > 0:
                numEdges[(node1, node2)] = G.number_of_edges(node1, node2)

# numEdges = sorted(numEdges.items(), key=lambda kv: kv[1])
# print(numEdges)

toRemove = []
for key in numEdges.keys():
    if numEdges[key] <= 1:
        G.remove_edge(key[0], key[1])
        toRemove.append((key[0], key[1]))
for key in toRemove:
    numEdges.pop(key)

keys = list(numEdges.keys())
dont = []
for key in keys:
    if key not in dont:
        a = numEdges[key]
        if (key[1], key[0]) in numEdges.keys():
            b = numEdges[(key[1], key[0])]
            if a / (a+b) >= .4 and a / (a+b) <= .6:
                for i in range(a):
                    G.remove_edge(key[0], key[1])
                for i in range(b):
                    G.remove_edge(key[1], key[0])
                dont.append((key[1], key[0]))
            elif a / (a+b) < .4:
                for i in range(a):
                    G.remove_edge(key[0], key[1])
                for i in range(b-1):
                    G.remove_edge(key[1], key[0])
                dont.append((key[1], key[0]))
            elif a / (a+b) > .6:
                for i in range(b):
                    G.remove_edge(key[1], key[0])
                for i in range(a-1):
                    G.remove_edge(key[0], key[1])
                dont.append((key[1], key[0]))
        else:
            for i in range(numEdges[key]-1):
                G.remove_edge(key[0], key[1])
#
nodes = list(G.nodes)
numEdges = {}
for node1 in nodes:
    for node2 in nodes:
        if node1 != node2:
            if G.number_of_edges(node1, node2) > 0:
                numEdges[(node1, node2)] = G.number_of_edges(node1, node2)
# print(numEdges)

for node in nodes:
    if G.number_of_edges(node, node) > 0:
        for i in range(G.number_of_edges(node, node)):
            G.remove_edge(node, node)

# print(G.edges())
G = nx.transitive_reduction(G)
# for x in nodes:
#     for y in nodes:
#         if x!=y:
#             max = -1
#             maxIndex = -1
#             paths = nx.all_simple_paths(G, x, y)
#             for i in range(len(paths)):
#                 if len(paths[i]) > max:
#                     max = len(paths[i])
#                     maxIndex = i
#             for i in range(len(paths)):
#                 if i != maxIndex:
#                     for j in range(len(paths[i])-1):
#                         # for k in range(k, len(paths[i])):
#                         G.remove_edge(paths[j], paths[j+1])

            # print(max(nx.all_simple_paths(G, x, y), key=lambda z: len(z)))



# for subgraph in nx.strongly_connected_component_subgraphs(G):
#     print(subgraph.nodes())

# import numpy
# nEdges=list(numEdges.values())
# nEdges.sort()
# mean = numpy.mean(nEdges, axis=0)
# sd = numpy.std(nEdges, axis=0)
# print(nEdges)
# print(mean)
# print(sd)
# final_list = [x for x in nEdges if (x > mean - 1 * sd)]
# # final_list = [x for x in final_list if (x < mean + 1 * sd)]
# print(final_list)

# numEdges = sorted(numEdges.items(), key=lambda kv: kv[1])
# print(numEdges)



# countDict = {}
# for key in numEdges.keys():
#     if numEdges[key] not in countDict.keys():
#         countDict[numEdges[key]] = 1
#     else:
#         countDict[numEdges[key]] += 1
# countDict = sorted(countDict.items(), key=lambda kv: kv[0])
# print(countDict)
#
# import xlsxwriter
# workbook = xlsxwriter.Workbook('arrays2.xlsx')
# worksheet = workbook.add_worksheet()
# row = 0
# col = 0
# for key, value in countDict:
#     worksheet.write(row, col, key)
#     worksheet.write(row, col+1, value)
#     row += 1
# workbook.close()
import random
from anytree import AnyNode, RenderTree
# tracesTree = nx.DiGraph()
starts = []
for node in G.nodes():
    if not G.in_edges(node):
        starts.append(node)

# tracesTree.add_node(0, value="Start", id=0)
startNode = AnyNode(id="Start")
startNodes = []
import copy
for node in starts:
    # tracesTree.add_node(len(tracesTree.nodes()), value=node, id=len(tracesTree.nodes()))
    # tracesTree.add_edge(0, len(tracesTree.nodes())-1)
    n = AnyNode(id=node, parent=startNode)
    startNodes.append(n)
    # startNodes.append(tracesTree.node[len(tracesTree.nodes())-1])
nodeCount = 4
for node in startNodes:
    closed = []
    openList = [node]
    exhaustedStates = []
    options = {}
    current = node
    # back = False
    while True:
        # if not back:
        old = copy.deepcopy(current)
        if current.id in options.keys():
            current = options[current.id].pop()
            if len(options[old.id]) == 0:
                options.pop(old.id)

        closed.append(current)
        # back = False

        if current.id not in options.keys():
            options[current.id] = []

            for node2 in G.nodes():
                inEdges = [x[0] for x in G.in_edges(node2)]
                c = [x.id for x in closed]
                if (set(inEdges).issubset(set(c))) and node2 != current.id and node2 not in c:
                    n = AnyNode(id=node2, parent=current)
                    nodeCount += 1
                    # openList.append(n)
                    if n not in options[current.id]:
                        options[current.id].append(n)
                    # numUnvisited += 1

        if len(options[current.id]) == 0:
            # back = True
            options.pop(current.id)
            if not list(options.keys()):
                break
            toStart = {}
            for key in options.keys():
                p = options[key][0].parent
                length = 1
                while p.id != "Start":
                    p=p.parent
                    length +=1
                toStart[key] = length
            o = max(toStart, key=toStart.get)
            current = options[o][len(options[o])-1].parent
            parentsOfCurrent = []
            parent = current.parent
            while parent:
                parentsOfCurrent.append(parent)
                parent = parent.parent

            toRemove = []
            for n in closed:
                if n.parent not in parentsOfCurrent:
                    toRemove.append(n)
            for r in toRemove:
                closed.remove(r)
    # while len(pastNodes) < len(G.nodes()) - 1:
    # for node2 in startNodes:
    #     if node2 != node:
    #         nextNodes.append(node2)
    # nodeCount = {
    #     'lower bridge':0,
    #     'key 1':0,
    #     'gem 2':0
    # }

    # while len(openList) > 0:
    #     current = openList.pop()
    #     numUnvisited = 0
    #     closed.append(current.id)
    #     if 'exit0' not in closed:
    #         for node2 in G.nodes():
    #             inEdges = [x[0] for x in G.in_edges(node2)]
    #             c = [st[:-1] for st in closed]
    #             if (set(inEdges).issubset(set(c))) and node2 not in closed:
    #                 if node2 in nodeCount.keys():
    #                     if node2+str(nodeCount[node2]) in closed or node2+str(nodeCount[node2]) == current.id:
    #                         continue
    #
    #                 if numUnvisited == 0:
    #                     if node2 in nodeCount.keys():
    #                         nodeCount[node2] += 1
    #                     else:
    #                         nodeCount[node2] = 0
    #                     n = AnyNode(id=node2+str(nodeCount[node2]), parent=current)
    #                     openList.append(n)
    #                 numUnvisited += 1
    #
    #     if numUnvisited == 0:
    #         if current.id not in exhaustedStates:
    #             exhaustedStates.append(current.id)
    #         toRemove = []
    #         for i in range(len(closed) - 1):
    #             if closed[len(closed) - 1 - i] not in exhaustedStates:
    #                 current = closed[len(closed) - 1 - i]
    #                 break
    #             toRemove.append(closed[len(closed) - 1 - i])
    #         for r in toRemove:
    #             closed.remove(r)
    #         for r in toRemove:
    #             if nodeCount[r[:-1]] > 0:
    #                 nodeCount[r[:-1]] -= 1
    #         exhaustedStates = []
    #
    #
    #     if numUnvisited == 1 and current.id not in exhaustedStates:
    #         exhaustedStates.append(current.id)



# from anytree.exporter import DotExporter
# DotExporter(startNode).to_picture("udo.png")
print(nodeCount)
from anytree.exporter import UniqueDotExporter
# for line in UniqueDotExporter(startNode, nodeattrfunc=lambda n: 'label="%s"' % (n.id)):
#     print(line)
UniqueDotExporter(startNode, nodeattrfunc=lambda n: 'label="%s"' % (n.id)).to_picture('test.png')
# print(RenderTree(startNode))

# write_dot(tracesTree, 'test.dot')
# pos = graphviz_layout(tracesTree, prog='dot')
# plt.title('draw_networkx')
# pos = hierarchy_pos(G, 1)
# plt.title('draw_networkx')
# pos = nx.planar_layout(tracesTree)
# nx.draw(tracesTree, node_size=1000, node_color='r', node_shape='s', with_labels=False)
# labels = {}

# for i in tracesTree:
#     labels[i] = tracesTree.node[i]['value']

# nx.draw_networkx_labels(tracesTree, pos, labels, font_size=12)
# plt.show()
# nx.draw(tracesTree, with_labels=True, font_weight='bold')
# plt.show()

# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()


    


# if __name__ == "__main__": main()
