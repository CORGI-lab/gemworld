import random
import math
from operator import add
from ast import literal_eval as make_tuple
import tracery
from tracery.modifiers import base_english
# import seq2seq.seq2seqbeamsearch
from anytree import AnyNode, PreOrderIter
from eventOrdering import startNode


rules = {
    'mvlev1P': ['I am #movement# towards lever 1 #flavor#.'],
    'mvlev2P': ['I am #movement# towards lever 2 #flavor#.'],
    'mvbridgeP': ['I am #movement# towards the bridge #flavor#.'],
    'overbridgeP': ['I am #cross# the bridge #flavor#.'],
    'lev1lowerP': ['I am #pull# lever 1 and lowering the bridge #flavor#.'],
    'lev1liftP': ['I am #pull# lever 1 and lifting the bridge #flavor#.'],
    'lev2lowerP': ['I am #pull# lever 2 and lowering the bridge #flavor#.'],
    'lev2liftP': ['I am #pull# lever  2 and lifting the bridge #flavor#.'],
    'mvsafeP': ['I am #movement# towards the safe spot #flavor#.'],
    'mvdoor1P': ['I am #movement# towards door 1 #flavor#.'],
    'mvdoor2P': ['I am #movement# towards door 2 #flavor#.'],
    'mvkey1P': ['I am #movement# towards key 1 #flavor#.'],
    'mvkey2P': ['I am #movement# towards key 2 #flavor#.'],
    'mvgem1P': ['I am #movement# towards the gem 1 #flavor#.'],
    'mvgem2P': ['I am #movement# towards the gem 2 #flavor#.'],
    'obkey1P': ['I am #obtainment# key 1 #flavor#.'],
    'obkey2P': ['I am #obtainment# key 2 #flavor#.'],
    'obgem1P': ['I am #obtainment# gem 1 #flavor#.'],
    'obgem2P': ['I am #obtainment# gem 2 #flavor#.'],
    'undoor1P': ['I am #unlock# door 1 #flavor#.'],
    'undoor2P': ['I am #unlock# door 2 #flavor#.'],
    'reachsafeP': ['I am #reach# the safe spot #flavor#.'],
    'exitP': ['I am #exitPt# #safe# #flavor#.'],
    'explore': ['I am just exploring #flavor#.'],
    'exploreF': ['I want to explore #flavor#.'],
    'death': ['I died #flavor#.'],
    'mvlev1F': ['I want to #moveF# towards lever 1 eventually.'],
    'mvlev2F': ['I want to #moveF# towards lever 2 eventually.'],
    'mvbridgeF': ['I want to #moveF# towards the bridge eventually.'],
    'overbridgeF': ['I want to #crossF# the bridge eventually.'],
    'lev1lowerF': ['I want to #pullF# lever 1 and lower the bridge eventually.'],
    'lev1liftF': ['I want to #pullF# lever 1 and lift the bridge eventually.'],
    'lev2lowerF': ['I want to #pullF# lever 2 and lower the bridge eventually.'],
    'lev2liftF': ['I want to #pullF# lever 2 and lift the bridge eventually.'],
    'mvsafeF': ['I want to #moveF# towards the safe spot eventually.'],
    'mvdoor1F': ['I want to #moveF# towards door 1 eventually.'],
    'mvdoor2F': ['I want to #moveF# towards door 2 eventually.'],
    'mvkey1F': ['I want to #moveF# towards key 1 eventually.'],
    'mvkey2F': ['I want to #moveF# towards key 2 eventually.'],
    'mvgem1F': ['I want to #moveF# towards gem 1 eventually.'],
    'mvgem2F': ['I want to #moveF# towards gem 2 eventually.'],
    'obkey1F': ['I want to #obtainF# key 1 eventually.'],
    'obkey2F': ['I want to #obtainF# key 2 eventually.'],
    'obgem1F': ['I want to #obtainF# gem 1 eventually.'],
    'obgem2F': ['I want to #obtainF# gem 2 eventually.'],
    'undoor1F': ['I want to #unlockF# door 1 eventually.'],
    'undoor2F': ['I want to #unlockF# door 2 eventually.'],
    'reachsafeF': ['I want to #reachF# the safe spot eventually.'],
    'exitF': ['I want to #exitFt# #safe# eventually.'],
    'movement': ['moving', 'heading', 'advancing', 'running', 'walking', 'travelling', 'going'],
    'obtainment': ['getting', 'obtaining', 'taking', 'grabbing', 'picking up', 'procuring', 'fetching'],
    'unlock': ['unlocking', 'opening', 'unlatching'],
    'reach': ['reaching', 'getting to', 'arriving at', 'entering', 'landing on'],
    'exitP': ['exiting', 'escaping', 'leaving', 'getting out'],
    'safe': ['unharmed', 'unhurt', 'safely', 'in one piece'],
    'flavor': ['with gusto', 'slowly but surely', 'in style'],
    'moveF': ['move', 'head', 'advance', 'run', 'walk', 'travel', 'go'],
    'obtainF': ['get', 'obtain', 'take', 'grab', 'pick up', 'procure', 'fetch'],
    'unlockF': ['unlock', 'open', 'unlatch'],
    'reachF': ['reach', 'get to', 'arrive at', 'enter', 'land on'],
    'exitFt': ['exit', 'escape', 'leave', 'get out'],
    'cross': ['crossing', 'going over'],
    'pull': ['pulling', 'yanking'],
    'crossF': ['cross', 'go over'],
    'pullF': ['pull', 'yank']
}

grammar = tracery.Grammar(rules)
grammar.add_modifiers(base_english)
ALPHA = .5  # learning rate dicatates how much values can swing at one time
GAMMA = .9  # discount factor defines how far ahead are you willing to look, make sure you prioritize short term rewards


# EPSILON=.2 #epsilon greedy (chance we randomly explore, .8 probability we execute an action)

class Grid:
    WALL1A = 5
    WALL1B = 4

    WALL2A = 4
    WALL2B = 5

    WALL3A = 4
    WALL3B = 3

    WALL4A = 7
    WALL4B = 6

    WALL5A = 7
    WALL5B = 8

    WALL6A = 8
    WALL6B = 7

    DEATH1A = 2
    DEATH1B = 4

    DEATH2A = 9
    DEATH2B = 9

    DEATH3A = 8
    DEATH3B = 8

    RIVER = 1

    BRIDGEA = 1
    BRIDGEB = 5

    LEVER1A = 0
    LEVER1B = 3

    LEVER2A = 2
    LEVER2B = 3

    SAFEA = 7
    SAFEB = 7

    GEM1A = 4
    GEM1B = 4

    GEM2A = 7
    GEM2B = 2

    KEY1A = 6
    KEY1B = 6

    KEY2A = 0
    KEY2B = 8

    DOOR1A = 3
    DOOR1B = 4

    DOOR2A = 6
    DOOR2B = 7

    BRIDGEDOWN = 0

    HASGEM1 = 0
    HASGEM2 = 0
    HASKEY1 = 0
    HASKEY2 = 0

    # Game logic:
    # walls-Bob cannot be on a gridspace where a wall is located(must go around walls but doesn't die)
    # safe-Grid has one safe spot where if you reach it and have the gem, you can exit safely. If reach the safe spot with no gem, nothing happens
    # death-Grid has 3 death spot where if you land on them you die and the game ends
    # gem-Bob must pick up the gem in order to exit
    # keys-there are 2 keys, one to unlock each door, must have picked up key1 to unlock door1
    # doors-Bob must travel around doors unless he has the key for the door. If he does, he can travel through them
    # river-takes up a while column of grid (Bob can't move around it). If Bob moves to a gridspace that is part of the river, he dies
    # bridge-Bob can only cross the river at the bridge
    # lever-Bob must stand on the lever to lower the bridge, can stand on it again to lift the bridge

    def __init__(self, width=10, height=10, initialVal=' '):
        self.width = width
        self.height = height
        # make grid of given height and given width
        self.data = [[initialVal for y in range(height)] for x in range(width)]
        # populate grid with obstacles and items
        self.data[self.WALL1A][self.WALL1B] = 'x'
        self.data[self.WALL2A][self.WALL2B] = 'x'
        self.data[self.WALL3A][self.WALL3B] = 'x'
        self.data[self.WALL4A][self.WALL4B] = 'x'
        self.data[self.WALL5A][self.WALL5B] = 'x'
        self.data[self.WALL6A][self.WALL6B] = 'x'
        self.data[self.SAFEA][self.SAFEB] = 't'
        self.data[self.DEATH1A][self.DEATH1B] = 'd'
        self.data[self.DEATH2A][self.DEATH2B] = 'd'
        self.data[self.DEATH3A][self.DEATH3B] = 'd'
        self.data[self.GEM1A][self.GEM1B] = 'G1'
        self.data[self.GEM2A][self.GEM2B] = 'G2'
        self.data[self.KEY1A][self.KEY1B] = 'k1'
        self.data[self.KEY2A][self.KEY2B] = 'k2'
        self.data[self.DOOR1A][self.DOOR1B] = 'd1'
        self.data[self.DOOR2A][self.DOOR2B] = 'd2'
        for i in range(height):
            self.data[self.RIVER][i] = 'r'
        self.data[self.LEVER1A][self.LEVER1B] = 'l1'
        self.data[self.LEVER2A][self.LEVER2B] = 'l2'
        self.wall = [self.WALL1A, self.WALL1B]
        self.endGood = [self.SAFEA, self.SAFEB]
        self.endBad = [self.DEATH1A, self.DEATH1B]
        self.gem1 = [self.GEM1A, self.GEM1B]
        self.gem2 = [self.GEM2A, self.GEM2B]
        self.key1 = [self.KEY1A, self.KEY1B]
        self.key2 = [self.KEY2A, self.KEY2B]
        self.door1 = [self.DOOR1A, self.DOOR1B]
        self.door2 = [self.DOOR2A, self.DOOR2B]
        self.lever1 = [self.LEVER1A, self.LEVER1B]
        self.lever2 = [self.LEVER2A, self.LEVER2B]

        # action[0]-move right
        # action[1]-move left
        # action[2]-move up
        # action[3]-move down
        # action[4]-pick up gem1 (can't drop items)
        # action[5]-pick up key1
        # action[6]-pick up key2
        # action[7]-open door1
        # action[8]-open door2
        # action[9]-pick up gem2
        # action[10]-lower bridge
        # action[11]-lift bridge


        # state[0][1]-(x,y) position of agent
        # state[2]-agent has gem1
        # state[3]-agent has key1
        # state[4]-agent has key2
        # state[5]-door1 open
        # state[6]-door2 open
        # state[7]-agent has gem2
        # state[8]-bridge is down

        self.action = [(1, 0, 0, 0, 0, 0, 0, 0, 0), (-1, 0, 0, 0, 0, 0, 0, 0, 0), (0, -1, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 0, -1)]

    # checks and sets grid information based on agent's inventory(found using agent's state) and the status of the bridge
    def checkForItem(self, state):
        # if agent is on gem, key1, or key2, tell grid that the agent has the gem, set agents inventory appropriately
        # set gem location to -1,-1 so its not on grid anymore and agent can't pick it up again
        if self.data[state[0]][state[1]] == 'd1':
            state = tuple(map(add, state, self.action[7]))
            self.door1[0] = -1
            self.door1[1] = -1
        elif self.data[state[0]][state[1]] == 'd2':
            state = tuple(map(add, state, self.action[8]))
            self.door2[0] = -1
            self.door2[1] = -1
        if self.data[state[0]][state[1]] == 'G1':
            state = tuple(map(add, state, self.action[4]))
            self.HASGEM1 = 1
            self.gem1[0] = -1
            self.gem1[1] = -1
            # print("HAS GEM 1")
        if self.data[state[0]][state[1]] == 'G2':
            state = tuple(map(add, state, self.action[9]))
            self.HASGEM2 = 1
            self.gem2[0] = -1
            self.gem2[1] = -1
            # print("HAS GEM 2")

        elif self.data[state[0]][state[1]] == 'k1':
            state = tuple(map(add, state, self.action[5]))
            self.HASKEY1 = 1
            self.key1[0] = -1
            self.key1[1] = -1
            # print("HAS KEY 1")
        elif self.data[state[0]][state[1]] == 'k2':
            state = tuple(map(add, state, self.action[6]))
            self.HASKEY2 = 1
            self.key2[0] = -1
            self.key2[1] = -1
            # print("HAS KEY 2")
        elif self.data[state[0]][state[1]] == 'l1' or self.data[state[0]][state[1]] == 'l2':
            # if agent on lever, if the bridge is up, change agent's state to bridge down, tell grid bridge is down, put bridge on grid
            if self.BRIDGEDOWN == 0:
                state = tuple(map(add, state, self.action[10]))
                self.BRIDGEDOWN = 1
                # print("BRIDGE DOWN")
                self.data[self.BRIDGEA][self.BRIDGEB] = 'b'
            # if bridge is down, change agent's state to bridge up, tell grid bridge is up, take bridge off of grid and replace with river
            elif self.BRIDGEDOWN == 1:
                state = tuple(map(add, state, self.action[11]))
                self.BRIDGEDOWN = 0
                # print("BRIDGE UP")
                self.data[self.BRIDGEA][self.BRIDGEB] = 'r'

        # as long as spot is not terminal spot
        if self.data[state[0]][state[1]] != 't' and self.data[state[0]][state[1]] != 'r' and self.data[state[0]][
            state[1]] != 'b' and self.data[state[0]][state[1]] != 'l1' and self.data[state[0]][state[1]] != 'l2':
            self.data[state[0]][state[1]] = 0
        return state

    def moveUp(self, state):
        # if you are not at the top of the grid
        if state[1] > 0:
            # if the next gridspace is not a wall or door, allow agent to move up freely
            if self.data[state[0]][state[1] - 1] != 'x' and self.data[state[0]][state[1] - 1] != 'd1' and \
                    self.data[state[0]][state[1] - 1] != 'd2':
                state = tuple(map(add, state, self.action[2]))
                state = self.checkForItem(
                    state)  # update state and inventory according to agent's position after the move
            # if the next gridspace is a door, if agent has corresponding key, or if door already open, allow them to move up freely and update state and inventory
            elif self.data[state[0]][state[1] - 1] == 'd1' and (state[3] == 1 or state[5] == 1):
                # print("OPEN DOOR 1")
                state = tuple(map(add, state, self.action[2]))
                state = self.checkForItem(state)
            elif self.data[state[0]][state[1] - 1] == 'd2' and (state[4] == 1 or state[6] == 1):
                # print("OPEN DOOR 2")
                state = tuple(map(add, state, self.action[2]))
                state = self.checkForItem(state)
        # else(if wall or door with no key, don't allow agent to move to that spot

        return state  # return the updated state after the move

    # repeat for other directions
    def moveDown(self, state):
        if state[1] < self.height - 1:
            if self.data[state[0]][state[1] + 1] != 'x' and self.data[state[0]][state[1] + 1] != 'd1' and \
                    self.data[state[0]][state[1] + 1] != 'd2':
                state = tuple(map(add, state, self.action[3]))
                state = self.checkForItem(state)
            elif self.data[state[0]][state[1] + 1] == 'd1' and (state[3] == 1 or state[5] == 1):
                # print("OPEN DOOR 1")
                state = tuple(map(add, state, self.action[3]))
                state = self.checkForItem(state)
            elif self.data[state[0]][state[1] + 1] == 'd2' and (state[4] == 1 or state[6] == 1):
                # print("OPEN DOOR 2")
                state = tuple(map(add, state, self.action[3]))
                state = self.checkForItem(state)
        return state

    def moveLeft(self, state):
        if state[0] > 0:
            if self.data[state[0] - 1][state[1]] != 'x' and self.data[state[0] - 1][state[1]] != 'd1' and \
                    self.data[state[0] - 1][state[1]] != 'd2':
                state = tuple(map(add, state, self.action[1]))
                state = self.checkForItem(state)
            elif self.data[state[0] - 1][state[1]] == 'd1' and (state[3] == 1 or state[5] == 1):
                # print("OPEN DOOR 1")
                state = tuple(map(add, state, self.action[1]))
                state = self.checkForItem(state)
            elif self.data[state[0] - 1][state[1]] == 'd2' and (state[4] == 1 or state[6] == 1):
                # print("OPEN DOOR 2")
                state = tuple(map(add, state, self.action[1]))
                state = self.checkForItem(state)
        return state

    def moveRight(self, state):
        if state[0] < self.width - 1:
            if (self.data[state[0] + 1][state[1]] != 'x') and self.data[state[0] + 1][state[1]] != 'd1' and \
                    self.data[state[0] + 1][state[1]] != 'd2':
                state = tuple(map(add, state, self.action[0]))
                state = self.checkForItem(state)
            elif self.data[state[0] + 1][state[1]] == 'd1' and (state[3] == 1 or state[5] == 1):
                # print("OPEN DOOR 1")
                state = tuple(map(add, state, self.action[0]))
                state = self.checkForItem(state)
            elif self.data[state[0] + 1][state[1]] == 'd2' and (state[4] == 1 or state[6] == 1):
                # print("OPEN DOOR 2")
                state = tuple(map(add, state, self.action[0]))
                state = self.checkForItem(state)
        return state

    # defines rewards for agent based on grid position
    def reward(self, state):
        # if agent is at the safe spot and has the gem
        if self.data[state[0]][state[1]] == 't' and state[2] == 1 and state[7] == 1:
            reward = 10  # reward agent +10
            # print("REWARD")
            # print()
            return state, reward
        # if agent has gem or key
        elif self.HASGEM1 == 1 or self.HASKEY1 == 1 or self.HASKEY2 == 1 or self.HASGEM2 == 1:
            reward = 1  # reward agent +1
            # set flags back to 0 so they can only get rewarded once(since you can only pick up and item once)
            self.HASGEM1 = 0
            self.HASGEM2 = 0
            self.HASKEY1 = 0
            self.HASKEY2 = 0
            return state, reward
        # if agent is on death spot or river
        elif self.data[state[0]][state[1]] == 'd' or self.data[state[0]][state[1]] == 'r':
            reward = -10  # reward agent -10
            return state, reward
        # any other spot in the grid
        else:
            reward = -1  # reward agent -1
            return state, reward


class Agent:
    EPSILON = .8
    count = 0

    def __init__(self, width, height):
        # initialize a grid for the agent to play on (initally all gridspaces will be 0)
        self.myGrid = Grid(initialVal=0)
        # initialize agent to be at pos (given x and y)
        self.pos = [width, height]
        # save inital position of agent
        self.originalPos = [width, height]
        # make empy dictionary for q table
        # key is agent state, as agent visits states, the q table will populate
        # value is array with 4 elements: element 0 is qtable value for moving up, etc
        self.qTable = {}
        self.reward = 0
        self.inventory = 0
        self.lastPos = [-1, -1]
        # set initial state to be agent current position, with no items and bridge up
        self.state = (self.pos[0], self.pos[1], 0, 0, 0, 0, 0, 0, 0)

    # reset grid and agent state in game so agent always starts in fresh grid with fresh state in each instance of the game
    def reset(self):
        self.myGrid = Grid(initialVal=0)
        self.pos = self.originalPos[:]
        self.state = (self.pos[0], self.pos[1], 0, 0, 0, 0, 0, 0, 0)
        self.reward = 0
        self.inventory = 0
        self.lastPos = [-1, -1]

    # dictates agent's movements
    def move(self, num):
        # for y in range(self.myGrid.height):
        # for x in range(self.myGrid.width):
        # print(self.myGrid.data[x][y],end=" ")
        # print()

        # move according to action we just chose, get and update state from the movement, get reward from movement, set the last action agent took to this action index
        if num == 0:
            self.state = self.myGrid.moveUp(self.state)
            # print("bob moves up")
            # print(self.state)
            # print()
            _, self.reward = self.myGrid.reward(self.state)
            self.lastAction = 0
        elif num == 1:
            self.state = self.myGrid.moveDown(self.state)
            # print("bob moves down")
            # print(self.state)
            # print()
            _, self.reward = self.myGrid.reward(self.state)
            self.lastAction = 1
        elif num == 2:
            self.state = self.myGrid.moveLeft(self.state)
            # print("bob moves left")
            # print(self.state)
            # print()
            _, self.reward = self.myGrid.reward(self.state)
            self.lastAction = 2
        elif num == 3:
            self.state = self.myGrid.moveRight(self.state)
            # print("bob moves right")
            # print(self.state)
            # print()
            _, self.reward = self.myGrid.reward(self.state)
            self.lastAction = 3
        else:
            print("rand.int was 4")
        # if agent has not visited this state (aka state not in qtable yet)
        if self.state not in self.qTable.keys():
            # initialize value of that state key in the table to array of zeros
            self.qTable[self.state] = [0.0, 0.0, 0.0, 0.0]

    # move randomly (so we discover as many possible states as possible)
    def moveRandom(self):
        num = random.randint(0, 3)
        self.move(num)

    # get the index of highest element in array corresponding to agent's current state
    # this will be the index of action that has the highest q value (best action to take) make best move for short term reward
    def moveGreedy(self):
        num = self.qTable[self.state].index(max(self.qTable[self.state]))
        self.move(num)

    def makeMove(self):
        self.count = self.count + 1
        if (self.count % 25 == 0):
            if self.EPSILON > .2:
                self.EPSILON = self.EPSILON - .25
        randnum = random.random()  # pick a random number between 0 and 1
        self.lastPos = self.pos[:]  # set last position to current position before moving
        self.lastAction = -1  # last action is none of the actions
        if randnum < self.EPSILON:
            self.moveRandom()
            # print("RANDOM")
        else:
            self.moveGreedy()
            # print("GREEDY")

    # initializes the q table to have one key and value pair of state at time of initialization and array of 0s
    def initQ(self):
        self.qTable[self.state] = [0.0, 0.0, 0, 0.0]


def main():
    # hidden_size = 256
    # encoder = seq2seq.seq2seqbeamsearch.EncoderRNN(seq2seq.seq2seqbeamsearch.inputState.n_words, hidden_size).to(
    #     seq2seq.seq2seqbeamsearch.device)
    # decoder = seq2seq.seq2seqbeamsearch.AttnDecoderRNN(hidden_size, seq2seq.seq2seqbeamsearch.outputState.n_words,
    #                                                    dropout_p=0.1).to(seq2seq.seq2seqbeamsearch.device)
    # model = seq2seq.seq2seqbeamsearch.torch.load("seq2seq/modelOnlyCurrent500000.tar")
    # encoder.load_state_dict(model['en'])
    # decoder.load_state_dict(model['de'])
    stateList = []
    # let bob play game 500 times
    # for x in range(10):
    #     if x == 0:
    #         bob = Agent(0, 0)  # set bob to start game at position 0,0 in grid
    #     elif x == 1:
    #         bob = Agent(3, 2)
    #     elif x == 2:
    #         bob = Agent(0, 6)
    #     elif x == 3:
    #         bob = Agent(3, 7)
    #     elif x == 4:
    #         bob = Agent(9, 0)
    #     elif x == 5:
    #         bob = Agent(8, 3)
    #     elif x == 6:
    #         bob = Agent(9, 8)
    #     elif x == 7:
    #         bob = Agent(7, 5)
    #     elif x == 8:
    #         bob = Agent(5, 3)
    #     elif x == 9:
    #         bob = Agent(4, 6)
    # bob = Agent(0, 0) #3000 #550 (not optimal)
    # bob = Agent(3, 2) #1100 #395 (optimal)
    # bob = Agent(0, 6) #2100 #420 (optimal)
    # bob = Agent(3, 7) #1400 #440 (optimal)
    # bob = Agent(9, 0) #850 #360 (optimal)
    # bob = Agent(8, 3) #900 #430 (optimal)
    # bob = Agent(9, 8) #1350 #400 (not optimal)
    # bob = Agent(7, 5) #1250 **** #405 (optimal)
    # bob = Agent(5, 3) #1100 #380 (optimal)
    bob = Agent(4, 6) #1100 #430 (optimal)
    bob.initQ()  # initialize q table
    for i in range(430):
        # reset reward to 0 every time a new game is played
        totalReward = 0
        # while bob is not in a state where he will die
        currentNode = startNode
        while bob.myGrid.data[bob.state[0]][bob.state[1]] != 'd' and bob.myGrid.data[bob.state[0]][bob.state[1]] != 'r':

            nextNodes = [node for node in PreOrderIter(startNode, filter_=lambda n: n.parent == currentNode)]
            nextNodesIds = [node.id for node in nextNodes]
            previousState = bob.state[:]  # set previous state to current state before moving
            bob.makeMove()
            lastAction = bob.lastAction  # get the last action bob took

            nodeIndex = -1
            if previousState[2] == 0 and bob.state[2] == 1:
                if "gem 1" in nextNodesIds:
                    nodeIndex = nextNodesIds.index("gem 1")

            if previousState[3] == 0 and bob.state[3] == 1:
                if "key 1" in nextNodesIds:
                    nodeIndex = nextNodesIds.index("key 1")

            if previousState[4] == 0 and bob.state[4] == 1:
                if "key 2" in nextNodesIds:
                    nodeIndex = nextNodesIds.index("key 2")

            if previousState[5] == 0 and bob.state[5] == 1:
                if "door 1" in nextNodesIds:
                    nodeIndex = nextNodesIds.index("door 1")

            if previousState[6] == 0 and bob.state[6] == 1:
                if "door 2" in nextNodesIds:
                    nodeIndex = nextNodesIds.index("door 2")

            if previousState[7] == 0 and bob.state[7] == 1:
                if "gem 2" in nextNodesIds:
                    nodeIndex = nextNodesIds.index("gem 2")

            if previousState[8] == 0 and bob.state[8] == 1:
                if "lower bridge" in nextNodesIds:
                    nodeIndex = nextNodesIds.index("lower bridge")

            if previousState[0] != 1 and previousState[1] != 5 and bob.state[0] == 1 and bob.state[1] == 5 and bob.state[8] == 1:
                if "cross bridge" in nextNodesIds:
                    nodeIndex = nextNodesIds.index("cross bridge")

            if nodeIndex != -1:
                bob.reward += 5
                currentNode = nextNodes[nodeIndex]
    #         # stateString = ""
    #         # for i in previousState:
    #         #     stateString += str(i) + " "
    #         # stateString += str(lastAction)
    #         # for i in bob.state:
    #         #     stateString += " " + str(i)
    #         # output_words, attentions, beams = seq2seq.seq2seqbeamsearch.evaluate(encoder, decoder, stateString)
    #         # policyInfoString = str(previousState) + ";" + str(lastAction) + ";" + str(bob.state)
    #         # for beam in beams:
    #         #     policyInfoString += ";" + beam
    #         # policyInfoString += ";\n"
    #         # with open("46/learningTraces.txt", "a+") as f:
    #         # 	f.write(policyInfoString)
    #         # 	f.close
            maxVal = max(bob.qTable[bob.state])  # get maximum value in q table for current state
            previousQ = bob.qTable[previousState][lastAction]  # get the q value of the previous (state, action) information (before the move)
    #
    #         # UNCOMMENT
    #         # possibleState = (previousState, lastAction, bob.state)
    #         # if possibleState not in stateList:
    #         #     stateList.append(possibleState)
    #         #     print("NEW STATE")
    #         # else:
    #         #     print("OLD STATE")
    #         # END UNCOMMENT
    #
            totalReward = totalReward + bob.reward  # keep track of the rewards bob is receiving while moving throughout the grid
            # calculate q value for current state and action and put it in q table
            bob.qTable[previousState][lastAction] = previousQ + ALPHA * (bob.reward + GAMMA * maxVal - previousQ)
            # if bob is at the safe spot and has the gem, stop the game
            if bob.myGrid.data[bob.state[0]][bob.state[1]] == 't' and bob.state[2] == 1 and bob.state[7] == 1:
                break
                # print("SAFE")
            elif bob.myGrid.data[bob.state[0]][bob.state[1]] == 'r' or bob.myGrid.data[bob.state[0]][bob.state[1]] == 'd':
                print("DEAD")
    #     # print(str(i)+": "+str(totalReward))
    #     # reset state and grid everytime start new game
        bob.reset()

    # print(bob.qTable)
    evaluatePolicy(bob)
    # getHierarchies(stateList, bob)

def evaluatePolicy(bob):
    # reset reward to 0
    print("EVAL POLICY")
    # hidden_size = 256
    # encoder = seq2seq.seq2seqbeamsearch.EncoderRNN(seq2seq.seq2seqbeamsearch.inputState.n_words, hidden_size).to(
    #     seq2seq.seq2seqbeamsearch.device)
    # decoder = seq2seq.seq2seqbeamsearch.AttnDecoderRNN(hidden_size, seq2seq.seq2seqbeamsearch.outputState.n_words,
    #                                                    dropout_p=0.1).to(seq2seq.seq2seqbeamsearch.device)
    # model = seq2seq.seq2seqbeamsearch.torch.load("seq2seq/modelOnlyCurrent500000.tar")
    # encoder.load_state_dict(model['en'])
    # decoder.load_state_dict(model['de'])
    totalReward = 0

    # with open("Traces/46/optimalPolicyActions.txt") as f:
    # 	actionsString = f.read()
    # for i in range(len(actionsString)-1):
    #     stateString = ""
    #     previousState = bob.state[:]
    #     bob.move(int(actionsString[i]))
    #     lastAction = bob.lastAction
    #     for i in previousState:
    #         stateString += str(i) + " "
    #     stateString += str(lastAction)
    #     for i in bob.state:
    #         stateString += " " + str(i)
    #     output_words, attentions, beams = seq2seq.seq2seqbeamsearch.evaluate(encoder, decoder, stateString)
    #     policyInfoString = str(previousState) + ";" + str(lastAction) + ";" + str(bob.state)
    #     # for beam in beams:
    #     #     policyInfoString += ";" + beam[0]
    #     policyInfoString += ";" + beams[0]
    #     stateString += " " + beams[0]
    #     if len(stateString.split(" ")) < 31:
    #         diff = 31 - len(stateString.split(" "))
    #         for j in range(diff):
    #             stateString += " " + str(-1)
    #
    #     stateString += "\n"
    #     policyInfoString += ";\n"
    #     with open("Traces/46/optimalTracesOnlyCurrent.txt ", "a+") as f:
    #     	f.write(stateString)
    # while bob is not in a state where he will die
    while bob.myGrid.data[bob.state[0]][bob.state[1]] != 'd' and bob.myGrid.data[bob.state[0]][bob.state[1]] != 'r':
        # stateString = ""
        # previousState = bob.state[:]
        # print(bob.state)
        bob.moveGreedy()
        print(bob.state)
        # lastAction = bob.lastAction
        totalReward = totalReward + bob.reward
        # with open("Traces/00/optimalPolicyActions.txt", "a") as f:
        #     f.write(str(lastAction))
        if bob.myGrid.data[bob.state[0]][bob.state[1]] == 't' and bob.state[2] == 1 and bob.state[7] == 1:
            print("SAFE")
            break
        elif bob.myGrid.data[bob.state[0]][bob.state[1]] == 'r' or bob.myGrid.data[bob.state[0]][bob.state[1]] == 'd':
            print("DEAD")
    # print(totalReward)


# get the distance between items and agent's current state
def moveTowards(state, bob, blacklist, hierarchy):
    boardLocations = [(bob.myGrid.GEM1A, bob.myGrid.GEM1B), (bob.myGrid.KEY1A, bob.myGrid.KEY1B), (bob.myGrid.KEY2A, bob.myGrid.KEY2B), (bob.myGrid.DOOR1A, bob.myGrid.DOOR1B), (bob.myGrid.DOOR2A, bob.myGrid.DOOR2B), (bob.myGrid.GEM2A, bob.myGrid.GEM2B), (bob.myGrid.BRIDGEA, bob.myGrid.BRIDGEB), (bob.myGrid.LEVER1A, bob.myGrid.LEVER1B), (bob.myGrid.LEVER2A, bob.myGrid.LEVER2B), (bob.myGrid.SAFEA, bob.myGrid.SAFEB)]
    distP = []
    distF = []
    gotCloser = []

    for i in range(len(boardLocations)):
        if i+2 < len(state[0]) - 1:
            if state[0][i+2] == 1 or state[2][i+2] == 1:
                distP.insert(i, 1000000)
                distF.insert(i, 1000000)
            else:
                distP.insert(i, math.sqrt((boardLocations[i][0]-state[0][0])**2+(boardLocations[i][1]-state[0][1])**2))
                distF.insert(i, math.sqrt((boardLocations[i][0]-state[2][0])**2+(boardLocations[i][1]-state[2][1])**2))
        else:
            distP.insert(i, math.sqrt((boardLocations[i][0] - state[0][0]) ** 2 + (boardLocations[i][1] - state[0][1]) ** 2))
            distF.insert(i, math.sqrt((boardLocations[i][0] - state[2][0]) ** 2 + (boardLocations[i][1] - state[2][1]) ** 2))

    for i in range(len(distP)):
        if distP[i] > distF[i] and i not in blacklist:
            gotCloser.append(distF[i])
        else:
            gotCloser.append(1000000)
    if min(gotCloser) != 1000000:
        moveTowardsHierarchy(state, bob, gotCloser.index(min(gotCloser)), blacklist, hierarchy)
        return

    else:
        moveTowardsHierarchy(state, bob, -1, blacklist, hierarchy)
        return

def moveTowardsHierarchy(state, bob, index, blacklist, hierarchy):
    gem1Index = 0
    key1Index = 1
    key2Index = 2
    door1Index = 3
    door2Index = 4
    gem2Index = 5
    bridgeIndex = 6
    lever1Index = 7
    lever2index = 8
    safeIndex = 9
    remaining = getRemaining(state)
    blacklistLength = len(blacklist)

    #need to deal with moving towards bridge and both levers

    sentences = [grammar.flatten("#mvgem1P#"), grammar.flatten("#mvkey1P#"), grammar.flatten("#mvkey2P#"), grammar.flatten("#mvdoor1P#"), grammar.flatten("#mvdoor2P#"), grammar.flatten("#mvgem2P#"), grammar.flatten("#mvbridgeP#"), grammar.flatten("#mvlev1P#"), grammar.flatten("#mvlev2P#"), grammar.flatten("#mvsafeP#")]
    if index != -1:
        if index == gem1Index:
            if key1Index+2 in remaining or door1Index+2 in remaining:
                blacklist.append(index)
        elif index == door1Index:
            if key1Index+2 in remaining:
                blacklist.append(index)
        elif index == door2Index:
            if key2Index+2 in remaining:
                blacklist.append(index)
        elif index == safeIndex:
            if gem1Index+2 in remaining or gem2Index+2 in remaining or door2Index+2 in remaining:
                blacklist.append(index)
        if len(blacklist) != blacklistLength:
            moveTowards(state, bob, blacklist, hierarchy)
        else:
            hierarchy.append(sentences[index])
            if index == bridgeIndex:
                hierarchy.append(grammar.flatten("#overbridgeF#"))
            if index == 7 or index == 8:
                #if moving towards lever 1 and bridge up
                if index == lever1Index and state[2][8] == 0:
                    hierarchy.append(grammar.flatten("#lev1lowerF#"))
                    hierarchy.append(grammar.flatten("#mvbridgeF#"))
                    hierarchy.append(grammar.flatten("#overbridgeF#"))
                    getRemainingHierarchy(remaining, hierarchy, 100)
                #if moving towards lever 1 and bridge down
                elif index == lever1Index and state[2][8] == 1:
                    hierarchy.append(grammar.flatten("#lev1liftF#"))
                    hierarchy.append(grammar.flatten("#exploreF#"))
                #if moving towards lever 2 and bridge up and has key 2
                elif index == lever2index and state[2][8] == 0 and state[2][4] == 1:
                    hierarchy.append(grammar.flatten("#lev2lowerF#"))
                    hierarchy.append(grammar.flatten("#exploreF#"))
                #if moving towards lever 2 and bridge up and no key 2
                elif index == lever2index and state[2][8] == 0 and state[2][4] == 0:
                    hierarchy.append(grammar.flatten("#lev2lowerF#"))
                    hierarchy.append(grammar.flatten("#mvbridgeF#"))
                    hierarchy.append(grammar.flatten("#overbridgeF#"))
                    getRemainingHierarchy(remaining, hierarchy, 100)
                elif index == lever2index and state[2][8] == 1:
                    hierarchy.append(grammar.flatten("#lev2liftF#"))
                    hierarchy.append(grammar.flatten("#exploreF#"))
            else:
                getRemainingHierarchy(remaining, hierarchy, index+2)
    else:
        hierarchy.append(grammar.flatten("#explore#"))


def getRemaining(x):
    remaining = []
    for i in range(2, 8):
        if x[0][i] == 0 and x[2][i] == 0:
            remaining.append(i)
    return remaining


def getRemainingHierarchy(remaining, hierarchy, startIndex):
    getKey1 = [grammar.flatten("#mvkey1F#"), grammar.flatten("#obkey1F#"), grammar.flatten("#mvdoor1F#"),
               grammar.flatten("#undoor1F#"), grammar.flatten("#mvgem1F#"), grammar.flatten("#obgem1F#")]
    getKey2 = [grammar.flatten("#mvkey2F#"), grammar.flatten("#obkey2F#"), grammar.flatten("#mvdoor2F#"),
               grammar.flatten("#undoor2F#"), grammar.flatten("#mvsafeF#")]

    import copy
    remainingCopy = copy.deepcopy(remaining)
    if startIndex < 8:
        if startIndex == 7:
            hierarchy.append(grammar.flatten("#obgem2F#"))
            remaining.remove(7)
        elif startIndex == 3:
            for i in range(len(getKey1)):
                if i != 0:
                    hierarchy.append(getKey1[i])
            remaining.remove(3)
            remaining.remove(5)
            remaining.remove(2)
        elif startIndex == 5:
            for x in range(3, 6):
                hierarchy.append(getKey1[x])
            remaining.remove(5)
            remaining.remove(2)
        elif startIndex == 2:
            hierarchy.append(getKey1[5])
            remaining.remove(2)
        elif startIndex == 4:
            for i in range(len(getKey2)):
                if i != 0 and i != len(getKey2) - 1: #so it doesn't say advancing towards safe spot in middle of hierarchy ????????????
                    hierarchy.append(getKey2[i])
            remaining.remove(4)
            remaining.remove(6)
        elif startIndex == 6:
            for x in range(3, 4): #(3,4) instead of (3,5) so it doesn't say advancing towards safe spot in middle of hierarchy ????????????
                hierarchy.append(getKey2[x])
            remaining.remove(6)

    if (7 in remaining):
        hierarchy.append(grammar.flatten("#mvgem2F#"))
        hierarchy.append(grammar.flatten("#obgem2F#"))
    if (3 in remaining):
        for x in getKey1:
            hierarchy.append(x)
    elif (5 in remaining):
        for x in range(2, 6):
            hierarchy.append(getKey1[x])
    elif (2 in remaining):
        for x in range(4, 6):
            hierarchy.append(getKey1[x])
    if (4 in remaining):
        for x in getKey2:
            hierarchy.append(x)
    elif (6 in remaining):
        for x in range(2, 5):
            hierarchy.append(getKey2[x])
    if len(remainingCopy) != 0 and getKey2[4] not in hierarchy and startIndex != 8:
        hierarchy.append(getKey2[4])


def getHierarchies(stateList, bob):
    trainingDict = {}
    # put all states agent visited into a file
    file = open("possibleStates2gems.txt", "w")
    file.write("\n".join(str(elem) for elem in stateList))
    file.close()
    with open('possibleStates2gems.txt') as file:
        stateList = file.readlines()
    file.close()

    key1SubH = [grammar.flatten("#obkey1P#"), grammar.flatten("#mvdoor1F#"), grammar.flatten("#undoor1F#"),
                grammar.flatten("#obgem1F#")]
    key2SubH = [grammar.flatten("#obkey2P#"), grammar.flatten("#mvdoor2F#"), grammar.flatten("#undoor2F#")]
    door1SubH = [grammar.flatten("#undoor1P#"), grammar.flatten("#mvgem1F#"), grammar.flatten("#obgem1F#")]

    subHierarchies = [[grammar.flatten("#obgem1P#")], key1SubH, key2SubH, door1SubH,
                          [grammar.flatten("#undoor2P#")], [grammar.flatten("#obgem2P#")]]
    # determine hierarchy of sentences for each state in the list of possible states (sentences only go forward in the hierarchy, do not explain the past)
    # trainingDict is dictionary with state as key and the hierarchy of sentences (as an array) for that state as the value
    # getAllStates(stateList)
    stateList = [x.strip() for x in stateList]
    for x in stateList:
        x = make_tuple(x)
        trainingDict[x] = []
        gridChanged = False

        if x[2][0] == bob.myGrid.SAFEA and x[2][1] == bob.myGrid.SAFEB and x[2][2] == 1 and x[2][7] == 1:
            trainingDict[x].append(grammar.flatten("#exitP#"))
            continue
        elif x[2][0] == bob.myGrid.DEATH1A and x[2][1] == bob.myGrid.DEATH1B or x[2][0] == bob.myGrid.DEATH2A and x[2][1] == bob.myGrid.DEATH2B or x[2][0] == bob.myGrid.DEATH3A and x[2][1] == bob.myGrid.DEATH3B or x[2][0] == 1:
            trainingDict[x].append(grammar.flatten("#death#"))
            continue
        else:
        # for everything but bridge
            for i in range(2, len(x[0])-1):
                if x[0][i] != x[2][i]:
                    for sent in subHierarchies[i - 2]:
                        trainingDict[x].append(sent)
                    gridChanged = True

            # if bridge state changed
            if x[0][8] != x[2][8]:
                gridChanged = True
                #if lowering it and on the right side of the river and key 2 has been picked up
                if x[2][8] == 1 and x[2][0] == 2 and x[2][4] == 1:
                    trainingDict[x].append(grammar.flatten("#lev2lowerP#"))
                    trainingDict[x].append(grammar.flatten("#exploreF#"))

                #if lifting it and on the right side of the river and key 2 has been picked up
                if x[2][8] == 0 and x[2][0] == 2 and x[2][4] == 1:
                    trainingDict[x].append(grammar.flatten("#lev2liftP#"))
                    trainingDict[x].append(grammar.flatten("#exploreF#"))

                # if lifting it and on left side of river
                elif x[2][8] == 0 and x[2][0] == 0:
                    trainingDict[x].append(grammar.flatten("#lev1liftP#"))
                    trainingDict[x].append(grammar.flatten("#exploreF#"))

                # if lowering it
                elif x[2][8] == 1:
                    #and on left side of river
                    if x[2][0] == 0:
                        trainingDict[x].append(grammar.flatten("#lev1lowerP#"))
                    #and on right side of river
                    elif x[2][0] == 2:
                        trainingDict[x].append(grammar.flatten("#lev2lowerP#"))

                    trainingDict[x].append(grammar.flatten("#mvbridgeF#"))
                    trainingDict[x].append(grammar.flatten("#overbridgeF#"))

                    remaining = getRemaining(x)
                    getRemainingHierarchy(remaining, trainingDict[x], 100)

            if not gridChanged:
                moveTowards(x, bob, [], trainingDict[x])

            isExploring = False
            for sent in trainingDict[x]:
                if "exploring" not in sent and "explore" not in sent:
                    isExploring = False
                else:
                    isExploring = True
            if not isExploring:
                trainingDict[x].append(grammar.flatten("#exitF#"))

    # 	#in a file, for each state write
    # 	#state1	hierarchySent1
    # 	#state1	hierarchySent2
    # 	#state2	hierarchySent1
    # 	#etc... (state and sent sep by tabs for future parsing)
    count = 0
    with open('trainingDict2gems.txt', 'w') as f:
        for key, value in trainingDict.items():
            for x in value:
                f.write('%s\t%s\n' % (key, x))
                count = count + 1

    # count the number of lines in the file
    print(count)
    f1 = open('trainingDict2gems.txt', 'r')
    f2 = open('training2gems.txt', 'w')
    #
    Lines = f1.readlines()
    # # print(Lines)
    i = 0
    lineArr = [];
    num = 0
    while num < (count * 80) / 100:
        lineArr.append(i)
        f2.write(Lines[i])
        i = random.randint(0, count - 1)
        while i in lineArr:
            i = random.randint(0, count - 1)
        num = num + 1
    # print(i)

    f1.close()
    f2.close()

def getHierarchy(x, bob):
    trainingDict = {}
    # put all states agent visited into a file

    key1SubH = [grammar.flatten("#obkey1P#"), grammar.flatten("#mvdoor1F#"), grammar.flatten("#undoor1F#"),
                grammar.flatten("#obgem1F#")]
    key2SubH = [grammar.flatten("#obkey2P#"), grammar.flatten("#mvdoor2F#"), grammar.flatten("#undoor2F#")]
    door1SubH = [grammar.flatten("#undoor1P#"), grammar.flatten("#mvgem1F#"), grammar.flatten("#obgem1F#")]

    subHierarchies = [[grammar.flatten("#obgem1P#")], key1SubH, key2SubH, door1SubH,
                          [grammar.flatten("#undoor2P#")], [grammar.flatten("#obgem2P#")]]
    # determine hierarchy of sentences for each state in the list of possible states (sentences only go forward in the hierarchy, do not explain the past)
    # trainingDict is dictionary with state as key and the hierarchy of sentences (as an array) for that state as the value
    # getAllStates(stateList)
    # x = make_tuple(x)
    trainingDict[x] = []
    gridChanged = False

    if x[2][0] == bob.myGrid.SAFEA and x[2][1] == bob.myGrid.SAFEB and x[2][2] == 1 and x[2][7] == 1:
        trainingDict[x].append(grammar.flatten("#exitP#"))
        return
    elif x[2][0] == bob.myGrid.DEATH1A and x[2][1] == bob.myGrid.DEATH1B or x[2][0] == bob.myGrid.DEATH2A and x[2][1] == bob.myGrid.DEATH2B or x[2][0] == bob.myGrid.DEATH3A and x[2][1] == bob.myGrid.DEATH3B or x[2][0] == 1:
        trainingDict[x].append(grammar.flatten("#death#"))
        return
    else:
    # for everything but bridge
        for i in range(2, len(x[0])-1):
            if x[0][i] != x[2][i]:
                for sent in subHierarchies[i - 2]:
                    trainingDict[x].append(sent)
                gridChanged = True

        # if bridge state changed
        if x[0][8] != x[2][8]:
            gridChanged = True
            #if lowering it and on the right side of the river and key 2 has been picked up
            if x[2][8] == 1 and x[2][0] == 2 and x[2][4] == 1:
                trainingDict[x].append(grammar.flatten("#lev2lowerP#"))
                trainingDict[x].append(grammar.flatten("#exploreF#"))

            #if lifting it and on the right side of the river and key 2 has been picked up
            if x[2][8] == 0 and x[2][0] == 2 and x[2][4] == 1:
                trainingDict[x].append(grammar.flatten("#lev2liftP#"))
                trainingDict[x].append(grammar.flatten("#exploreF#"))

            # if lifting it and on left side of river
            elif x[2][8] == 0 and x[2][0] == 0:
                trainingDict[x].append(grammar.flatten("#lev1liftP#"))
                trainingDict[x].append(grammar.flatten("#exploreF#"))

            # if lowering it
            elif x[2][8] == 1:
                #and on left side of river
                if x[2][0] == 0:
                    trainingDict[x].append(grammar.flatten("#lev1lowerP#"))
                #and on right side of river
                elif x[2][0] == 2:
                    trainingDict[x].append(grammar.flatten("#lev2lowerP#"))

                trainingDict[x].append(grammar.flatten("#mvbridgeF#"))
                trainingDict[x].append(grammar.flatten("#overbridgeF#"))

                remaining = getRemaining(x)
                getRemainingHierarchy(remaining, trainingDict[x], 100)

        if not gridChanged:
            moveTowards(x, bob, [], trainingDict[x])

        trainingDict[x].append(grammar.flatten("#exitF#"))

    print(trainingDict[x])

if __name__ == "__main__": main()
