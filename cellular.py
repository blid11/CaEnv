#Environment for RL agent, compatible with Kenny Young's Mini Atar DQN

import numpy as np
from Simulation import Simulation
import math

class Env:
    def __init__(self, ramping=None, seed=None):
        self.channels = {
            'y0': 0,
            'y1': 1,
            'y2': 2,
            'y3': 3,
            'y4': 4,
            'agent0': 5,
            'agent1': 6,
            'agent2': 7,
            'agent3': 8,
            'agent4': 9
        }
        self.action_map = ['n', 'l', 'u', 'r', 'd', 'f']
        self.random = np.random.RandomState(seed)
        self.signature_list = {}
        # self.signature_list = []
        # self.simRewCalc = Simulation()
        self.reset()

    def act(self, a):

        # print(self.agent_step)

        if self.agent_step == 3:
            self.agent_step = 0

        self.simRewCalc.resetSim()

        self.r = 0

        # print("it's working!")

        #start with num pieces <12 to see if it can at least make the larger glider
        #reward system is based on a lot of cells
        if self.num_pieces_placed > 30:
            self.terminal = True
            return self.r, self.terminal

        xcopy = self.x
        ycopy = self.y
        zcopy = self.z

        fire = False

        a = self.action_map[a]

        if a == 'u':
            # up
            xcopy = self.x
            ycopy = self.y
            zcopy = self.z + 1

        elif a == "d":
            # down
            xcopy = self.x
            ycopy = self.y
            zcopy = self.z - 1

        elif a == 'l':
            # left
            xcopy = self.x - 1
            ycopy = self.y
            zcopy = self.z

        elif a == 'r':
            # right
            xcopy = self.x + 1
            ycopy = self.y
            zcopy = self.z

        elif a == 'n':
            # "forward" , really backward
            xcopy = self.x
            ycopy = self.y - 1
            zcopy = self.z

        else:
            # choice == 5
            # fire = True
            # agent stays in position
            xcopy = self.x
            ycopy = self.y
            zcopy = self.z

        # If we are out of bounds, fix!
        # should a penalty be applied for going out of bounds?
        if xcopy < 0:
            xcopy = 0
            # self.reward = 0
        elif xcopy > 9:
            xcopy = 9

        if ycopy < 0:
            ycopy = 0
        elif ycopy > 4:
            ycopy = 4

        if zcopy < -5:
            zcopy = -5

        elif zcopy > 4:
            zcopy = 4

        # need to make sure that we are not moving to a cell that has already been moved to
        # if this occurs, negative reward and return to original position
        self.x = xcopy
        self.y = ycopy
        self.z = zcopy
        editCoord = (xcopy, ycopy, zcopy)

        if self.agent_step == 2:
            # the agent is trying to place a cell
            # first check if this cell has already been placed
            if editCoord in self.cellsToSimulatePre:
                # agent has placed a piece where there already is one
                # observation remains the same
                # self.x, self.y, self.z remain the same
                self.r = 0
                self.agent_step += 1
                return self.r, self.terminal

            else:

                # if a move is declared okay, it is mirrored and translated, then added to the list of
                # cells to simulate
                self.num_pieces_placed += 1

                if self.y == 4:
                    mirror = 5
                elif self.y == 3:
                    mirror = 6
                elif self.y == 2:
                    mirror = 7
                elif self.y == 1:
                    mirror = 8
                else:
                    mirror = 9

                self.unMirroredCells.append(editCoord)

                mirrorCoord = (self.x, mirror, self.z)

                self.cellsToSimulatePre.append(mirrorCoord)

                # for cell in self.cellsToSimulatePre:
                #     print("cell b4 sim:, {}".format(cell))

                self.cellsToSimulatePre.append(editCoord)

                # self.cellsToSimulate.clear()

                # translate cellsToSimulate, either to a corner or the centre of the grid
                # will translate all cells with a single vector e.g.(42, 42, 42)

                for cell in self.cellsToSimulatePre:
                    # this will include the original starting cell
                    xcoord2 = cell[0] + 60
                    ycoord2 = cell[1] + 60
                    zcoord2 = cell[2] + 60
                    indices2 = (xcoord2, ycoord2, zcoord2)
                    self.cellsToSimulate.append(indices2)

                self.cellsToSimulate.append(self.starting_position)
                self.cellsToSimulate.append(self.starting_position_mirror)

                # simulation is run to give a new observation, reward, done
                self.penalty = 0
                self.run_Simulation()
                self.agent_step += 1
                return self.r, self.terminal

        else:
            # the agent is not placing a cell and is only moving
            # the state will be updated to have self.x, self.y, self.z
            # the state is no longer just unMirrored, but also the position of the agent
            self.r = 0
            self.agent_step += 1
            return self.r, self.terminal


    def run_Simulation(self):
        # now with the mirrored/duplicated and translated cells, will run the simulation to calculate reward
        # the reward will depend on if a contiguous pattern of cells has hit a boundary
        # will find the most extreme cell and see how many cells are neighbours/neighbours of neighbours of it
        # this will determine the score
        # this will be the cell that has the greatest distance from the centre cell (sqrt formula)
        # this will be calculated at each time step for the list of live cells

        # self.simRewCalc.resetSim()

        # for cell in self.cellsToSimulate:
        #     print("cell b4 sim:, {}".format(cell))

        self.simRewCalc.ca_beg_to_end(self.cellsToSimulate)
        self.gliderV = self.simRewCalc.getGliderV()

        # print(len(self.cellsToSimulate))

        # if len(self.cellsToSimulate) == 10:
        #      self.simRewCalc.render()

        # this results in a cumulatively very negative reward, but the agent should try to optimize
        self.r = self.simRewCalc.getReward()

        signature = self.simRewCalc.return_signature()
        signature = tuple(signature)

        if sum(signature) == 0:
            self.r = 0

        elif signature in self.signature_list.keys():
            # it isn't generating anything new
            # penalize within the episode: makes the most sense
            # exponential decay according to value "repeated"
            # self.repeated += 1
            self.signature_list[signature] += 1

            # self.r = self.r * math.exp(-0.2*self.repeated)

            if self.r == 4 and self.repeated > 12:
                self.r = 0
            elif self.r == 4:
                self.r = math.exp(-3 * (self.signature_list[signature]) / (sum(self.signature_list.values())))
                self.repeated += 1
            else:
                self.r = math.exp(-3 * (self.signature_list[signature])/(sum(self.signature_list.values())))

        else:
            # self.signature_list.append(signature)
            self.signature_list[signature] = 1

        #once the agent gets a 1 or a 2, it cannot receive a reward again

        # if (self.num_pieces_placed > 7) and (self.r not in self.rewardList):
        #     #to avoid farming, introduce a bonus for mixing it up
        #     self.rewardList.append(self.r)
        #     self.r = self.r + 3

        # if self.r in self.rewardList:
        #     self.r = 0
        #     #to avoid farming, introduce a bonus for mixing it up
        # else:
        #     self.rewardList.append(self.r)

        self.total_score += self.simRewCalc.getReward()

        # if self.num_pieces_placed % 3 == 0:
        #     self.printEpReward()

    def get_total_score(self):
        return self.total_score

    def print_total_score(self):
        print("total score {}".format(self.total_score))

    def printEpReward(self):
        # if self.num_pieces_placed > 5:
        print("Agent reward for move:, {}, step: {}".format(self.r, self.num_pieces_placed))

    # Query the current level of the difficulty ramp, difficulty does not ramp in this game, so return None
    def difficulty_ramp(self):
        return None

        # Process the game-state into the 10x10xn state provided to the agent and return

    def state(self):
        state = np.zeros((10, 10, len(self.channels)), dtype=bool)

        position = (self.x, self.y, self.z)

        # recall the agent was allowed to place cells between x: 0 to 9 y: 0 to 4 and z: -5 to 4

        editCells = []

        for cell in self.unMirroredCells:
            ex = cell[0]
            why = cell[1]
            zed = cell[2] + 5
            coord = (ex, why, zed)
            editCells.append(coord)

        zholder = position[2] + 5

        #the position of the agent's y is increased by 5 so 0 is 5, 1 is 6, 2 is 7.. 4 is 9 so that the agent's and cells positions can be known
        newposition = (position[0], zholder, position[1] + 5)

        # the agent will view its drawn structure looking at the x - z plane, where y is depth
        for cell in editCells:
            x = cell[0]
            y = cell[1]
            z = cell[2]
            state[x][z][y] = 1

       #agent gets crosshairs that let it know where the timer is
        if self.agent_step == 2:
            state[newposition[0]][newposition[1]][newposition[2]] = 1

        elif self.agent_step == 1:
            if newposition[0] == 0:
                x2 = newposition[0]
                x3 = newposition[0] + 1
                state[x2][newposition[1]][newposition[2]] = 1
                state[x3][newposition[1]][newposition[2]] = 1
            elif newposition[0] == 9:
                x1 = newposition[0] - 1
                x2 = newposition[0]
                state[x2][newposition[1]][newposition[2]] = 1
                state[x1][newposition[1]][newposition[2]] = 1
            else:
                x1 = newposition[0] - 1
                x2 = newposition[0]
                x3 = newposition[0] + 1
                state[x2][newposition[1]][newposition[2]] = 1
                state[x1][newposition[1]][newposition[2]] = 1
                state[x3][newposition[1]][newposition[2]] = 1

        elif self.agent_step == 0:
            if newposition[2] == 0:
                y2 = newposition[2]
                y3 = newposition[2] + 1
                state[newposition[0]][newposition[1]][y2] = 1
                state[newposition[0]][newposition[1]][y3] = 1
            elif newposition[2] == 9:
                y2 = newposition[2]
                y1 = newposition[2] - 1
                state[newposition[0]][newposition[1]][y2] = 1
                state[newposition[0]][newposition[1]][y1] = 1
            else:
                y1 = newposition[2] - 1
                y2 = newposition[2]
                y3 = newposition[2] + 1
                state[newposition[0]][newposition[1]][y1] = 1
                state[newposition[0]][newposition[1]][y2] = 1
                state[newposition[0]][newposition[1]][y3] = 1

        return state

    # Reset to start state for new episode
    def reset(self):
        self.simRewCalc = Simulation()

        self.starting_position = (64, 64, 64)
        self.starting_position_mirror = (64, 65, 64)

        # these will be the indices of the cell the agent starts at
        self.x = 4
        self.y = 4
        self.z = 4

        self.total_score = 0

        # cells to simulate should have the first cell address
        self.unMirroredCells = [(4, 4, 4)]
        self.cellsToSimulatePre = []
        self.cellsToSimulate = []
        self.r = 0
        self.max_r = 0
        self.terminal = False
        self.num_pieces_placed = 0

        self.agent_step = 0

        self.rewardList = []

        # self.signature_list = []
        self.repeated = 0

        # self.terminal = False
        # # could also just recall the init function?
        # # is there a clear function for both these lists?
        # self.cellsToSimulate = []
        # self.cellsToSimulatePre = []
        # # cells to simulate should have the first cell address
        # self.x = 4
        # self.y = 4
        # self.z = 4
        #
        # self.num_pieces_placed = 0
        #
        # self.unMirroredCells = [(4, 4, 4)]
        #
        # self.simRewCalc.resetSim()

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10, 10, len(self.channels)]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ['n', 'l', 'u', 'r', 'd', 'f']
        return [self.action_map.index(x) for x in minimal_actions]
