import math

import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D


class Simulation:

    def __init__(self):

        self.starting_position = 64

        self.numLiveCells = 0

        self.generation = 1

        self.imm_Rew = 0
        self.avg_pop = 0

        self.hitLastBorder = False
        self.stillLife = False
        self.allDead = False
        self.oscillator = False
        self.gliderV = False

        # list of tuples (indices of cells)
        self.liveCellCurrent = []
        self.liveCellPrev = []

        # keep track of live cells and their neighbours, if the number of neighbours is 5, 6, or 7
        # if a cell goes from these values to outside of this range check liveCellsCurrent and kick em
        # if a cell goes from outside these values to 6, add em
        self.changelist_n_counts = {}

        self.contigList = []

        # MAKE BETTER VALUES

        self.populationList = []
        self.nonContigPop = []
        self.current_reward = 0
        self.total_reward = 0
        self.most_extreme_dist = 0
        self.current_most_extreme_cell = (self.starting_position, self.starting_position, self.starting_position)
        self.mightBeOsc = False

        self.stopwatch = 0

        self.dead_vicinity = {}
        self.live_vicinity = {}
        self.dead_hist = []
        for i in range(1, 27):
            self.dead_hist.append(0)
        self.live_hist = []
        for i in range(0, 27):
            self.live_hist.append(0)
        self.signature = []

    def get_starting_position(self):
        return self.starting_position

    def ca_beg_to_end(self, cellsToSimulate):

        self.liveCellCurrent = cellsToSimulate
        self.numLiveCells = len(cellsToSimulate)
        # perform the ca simulation from beginning (bring in the chosen cells)
        # to the end (simulation reaches an end condition)
        # alternate between generating neighbour counts for live cells, and then updating the live cell list:
        # start with ca_count_neigh, then ca_update_live, and alternate until ca_update_live:
        # contains 0 cells, stillLife (liveCellCurrent = prev), lasts for 100 generations w/ most extreme cell not changing
        # or the fourth condition: the outer border is hit

        # for cell in self.liveCellCurrent:
        #     print("cell during sim:, {}".format(cell))

        # call ca_count_neigh to get the first neighbour count
        self.ca_count_neigh()

        # have the reward set to the starting condition

        # (not allDead and not stillLife and not oscillator and not hitLastBorder)

        while not self.allDead and not self.stillLife and not self.oscillator and not self.hitLastBorder:
            # while self.generation < 1:
            self.ca_update_live()
            self.numLiveCells = len(self.liveCellCurrent)
            self.find_extreme_pattern()
            #population list is the number of contiguous cells
            self.nonContigPop.append(self.numLiveCells)
            self.populationList.append(len(self.contigList))

            # calculate the reward gained through this generation of the simulation (the sum of the rewards of a full simulation will be given to an agent for one action)
            # self.reward_calc()


            # print("gen, {}. live cells, {}".format(self.generation, self.numLiveCells))

            # compute reward, linked to hitLastBorder, extremeCell
            # extremeCell is also linked to whether or not you have an oscillator
            if self.numLiveCells == 0:
                # simulation done, rather than break
                self.allDead = True
                # unstable object can end in all cells dying, still give cumulative reward for how far it got
                self.reward_calc()
                self.total_reward = self.imm_Rew
                self.calc_signature()
                # print("all cells died")
                # print(self.signature)

                if self.total_reward > 4:
                    print("all cells died")
                    print("move reward: {}, numCells:  {}, maxDist:  {}, avg_cells: {}".format(self.total_reward,
                                                                                           len(self.contigList),
                                                                                           self.most_extreme_dist,
                                                                                           self.avg_pop))
                # print("all cells died")
                # print("move reward: {}, numCells:  {}, maxDist:  {}, avg_cells: {}".format(self.total_reward, len(self.contigList),
                #                                                                 self.most_extreme_dist, self.avg_pop))
                # print(self.nonContigPop)
                # print(self.populationList)
                # print("\n")
                # print(self.generation)
                # print(len(self.populationList))
                # print("\n")
                # reward = -300
            elif self.liveCellCurrent == self.liveCellPrev:
                self.stillLife = True
                self.reward_calc()
                self.total_reward = self.imm_Rew
                self.calc_signature()
                # print("still life")
                # print(self.signature)

                if self.total_reward > 4:
                    print("still life")
                    print("move reward: {}, numCells:  {}, maxDist:  {}, avg_cells: {}".format(self.total_reward,
                                                                                               len(self.contigList),
                                                                                               self.most_extreme_dist,
                                                                                               self.avg_pop))
                    #print(self.contigList)
                # if len(self.contigList) == 0:
                #   print(self.liveCellCurrent)
                #   self.render()
                # print("still life")
                # print("move reward: {}, numCells:  {}, maxDist:  {}, avg_cells: {}".format(self.total_reward,
                #                                                                            len(self.contigList),
                #                                                                            self.most_extreme_dist,
                #                                                                            self.avg_pop))
                # print(self.nonContigPop)
                # print(self.populationList)
                # print("\n")
                # print(self.generation)
                # print(len(self.populationList))
                # print("total reward:, {}".format(self.total_reward))
                # print("\n")
                # give a default reward, better than allDead
                # reward = -200?
            else:
                # get the current most extreme cell to find if we have an oscillator or if the pattern is evolving
                # self.find_extreme_pattern()

                if self.oscillator:
                    # this condition may not be true immediately but the agent does not know that, it does not care about the simulation
                    # while the most_extreme_cell is changing the reward is accumulated, if it does not change for more than 20 generations (timer == 20), that 20 generation gap
                    # does not accumulate the reward when an oscillating pattern was reached, the reward accumulated before 20 generations in the past is used for reward
                    self.reward_calc()
                    self.total_reward = self.imm_Rew
                    self.calc_signature()
                    # print("oscillator")
                    # print(self.signature)

                    if self.total_reward > 4:
                        print("oscillator")
                        print("move reward: {}, numCells:  {}, maxDist:  {}, avg_cells: {}".format(self.total_reward,
                                                                                                len(self.contigList),
                                                                                                self.most_extreme_dist,
                                                                                                self.avg_pop))
                        #print(self.contigList)
                    # print(self.nonContigPop)
                    # print(self.populationList)
                    # print("\n")
                    # print(self.generation)
                    # print(len(self.populationList))
                    # print("total reward:, {}".format(self.total_reward))
                    # print("\n")
                else:
                    # the agent receives the total cumulative reward for however far the pattern got
                    # since the pattern did not end in death, still life or oscillation it should be a stable pattern
                    # allow it to continue propagating until distance from the centre > 50
                    xcoord = self.current_most_extreme_cell[0]
                    ycoord = self.current_most_extreme_cell[1]
                    zcoord = self.current_most_extreme_cell[2]
                    if self.calc_dist(xcoord, ycoord, zcoord) > 50:
                        #to make the reward function more smooth, could subtract 23.. but then this removes incentive if a stable pattern is stumbled upon by accident
                        self.reward_calc()
                        self.total_reward = self.imm_Rew #- 13 #13 point bonus, -23 was -10 # need to counteract 22*30 is 660
                        self.hitLastBorder = True
                        self.calc_signature()
                        # print(self.signature)

                        if(len(self.contigList) == 10):
                            print("Glider V")
                        else:
                            print("stable object")
                            print(self.liveCellCurrent)
                            # print("\n")
                            print(self.contigList)
                            # file1 = open("newobj.txt", "a")
                            # sigString = " ".join(str(x) for x in self.contigList)
                            # file1.write(toString(self.contigList))
                            # file1.close
                            # print("\n")
                            print("move reward: {}, numCells:  {}, maxDist:  {}, avg_cells: {}".format(self.total_reward,
                                                                                                       len(self.contigList),
                                                                                                       self.most_extreme_dist,
                                                                                                       self.avg_pop))
                        # print(self.nonContigPop)
                        # print(self.populationList)
                        # print("\n")
                        # print(self.generation)
                        # print(len(self.populationList))
                        # # print(self.rewardList)
                        # print("total reward:, {}".format(self.total_reward))

            #populationList(0) should be equal to generation, will not include first state
            #this is to get the average number of cells

            self.generation += 1
            self.contigList.clear()
            self.ca_count_neigh()

    def ca_count_neigh(self):
        # this method only tallies the number of neighbours of liveCellCurrent in changelist_n_count

        self.changelist_n_counts.clear()

        for cell in self.liveCellCurrent:
            # calculate the neighbours of the cell and add them to the list

            # if cell not in self.changelist_n_counts.keys():
            #     #need to include the cell itself in the changelist if it has not already been added
            #     #actually this should not matter as it is only relevant if it has a live neighbour
            #     self.changelist_n_counts[cell] = 0

            xcoord = cell[0]
            ycoord = cell[1]
            zcoord = cell[2]

            for x in range(-1, 2):
                for y in range(-1, 2):
                    for z in range(-1, 2):
                        neighbour = (xcoord + x, ycoord + y, zcoord + z)
                        if x == 0 and y == 0 and z == 0:
                            # problem before was that if a live cell was already added, you reset the value to 0
                            if neighbour not in self.changelist_n_counts.keys():
                                self.changelist_n_counts[neighbour] = 0
                            else:
                                next
                            # this may resolve a problem, certain live cells are not being removed from liveCellCurrent
                            # yep it's because they end up having no live neighbours and so they are not added to the list
                            # not even to be removed
                            # this could be because they are not in changelistncounts
                            # self.changelist_n_counts[neighbour] = 0
                            # next
                        elif neighbour in self.changelist_n_counts.keys():
                            # the cell has already been added, increase value by 1
                            self.changelist_n_counts[neighbour] = self.changelist_n_counts[neighbour] + 1
                        else:
                            # else add the tuple as a key
                            self.changelist_n_counts[neighbour] = 1

    def ca_update_live(self):
        # set liveCellPrev = liveCellCurrent
        self.liveCellPrev = list(self.liveCellCurrent)

        # for cell in self.liveCellCurrent:
        #     print(cell)
        #
        # print("\n")

        # iterate over all relevant cell neighbour counts and determine the new liveCellCurrent
        # born = 0
        # died = 0

        for cellKey in self.changelist_n_counts:
            cell_n_count = self.changelist_n_counts[cellKey]
            # conditions if cell was alive or dead, depending on its live neighbour count
            # is not in liveCellPrev & state is 6: add to liveCellCurrent
            if (cellKey not in self.liveCellCurrent) and (cell_n_count == 6):
                self.liveCellCurrent.append(cellKey)
                # born +=1
                # print("{},cell born, {}".format(cell_n_count, cellKey))
            elif (cellKey in self.liveCellCurrent) and ((cell_n_count < 5) or (cell_n_count > 7)):
                self.liveCellCurrent.remove(cellKey)
                # died += 1
                # print("{},cell died, {}".format(cell_n_count, cellKey))
            # elif (cellKey in self.liveCellCurrent) and ((cell_n_count == 5) or (cell_n_count == 7) or (cell_n_count == 6)):
            #     # print("{},cell stay, {}".format(cell_n_count, cellKey))

        # print("{} cells in prev, {} in current, curr should be {}".format(len(self.liveCellPrev), len(self.liveCellCurrent), (len(self.liveCellPrev) + born - died)))

        # print("\n")
        #
        # for cell in self.liveCellCurrent:
        #     print(cell)
        #

    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for cell in self.liveCellCurrent:
            xs = cell[0]
            ys = cell[1]
            zs = cell[2]
            ax.scatter(xs, ys, zs)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    def find_extreme_pattern(self):
        # need to find the cell that is furthest from the centre area
        # and also find the number of cells that form a contiguous pattern with it

        # will go through the list of cells and calculate the distances between the x,y,z coord relative to starting position
        # if a cell is found to be >70 units away from the starting position, this is considered the final border
        maxDist = 0
        dist = 0
        for cell in self.liveCellCurrent:
            x = cell[0]
            y = cell[1]
            z = cell[2]
            dist = self.calc_dist(x, y, z)
            if dist > self.most_extreme_dist:
                # the pattern is moving further than it has before
                self.most_extreme_dist = dist
                self.current_most_extreme_cell = cell
                maxDist = dist
                self.mightBeOsc = False
            elif dist > maxDist:
                # the pattern has not moved further but we need to know its most extreme cell
                maxDist = dist
                self.current_most_extreme_cell = cell
                self.mightBeOsc = True
            else:
                self.mightBeOsc = True

        # now with the current most extreme cell, check if this value has changed since prev gen and if timer should be started
        self.check_if_oscillator()

        # with the most extreme cell, find how many cells form a contiguous pattern with it to calculate the reward
        self.contig_pattern()

    def contig_pattern(self):

        # find all cells that form a contiguous pattern, add them to contiglist

        # for each neighbour found, want to also investigate its neighbours recursively

        xcoord = self.current_most_extreme_cell[0]
        ycoord = self.current_most_extreme_cell[1]
        zcoord = self.current_most_extreme_cell[2]

        # check if the most_extreme_cell has neighbours
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    neighbour = (xcoord + x, ycoord + y, zcoord + z)
                    if x == 0 and y == 0 and z == 0:
                        next
                    elif neighbour in self.liveCellCurrent:
                        # add to contiglist if it is not there
                        if neighbour not in self.contigList:
                            self.contigList.append(neighbour)
                            # call contiguous on this live neighbour to perform this same loop on each of its neighbours
                            self.contiguous(neighbour)
                        else:
                            next
                    else:
                        # neighbour is not alive
                        next

    def contiguous(self, neighbour):
        xcoord = neighbour[0]
        ycoord = neighbour[1]
        zcoord = neighbour[2]

        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    neighbour = (xcoord + x, ycoord + y, zcoord + z)
                    if x == 0 and y == 0 and z == 0:
                        next
                    elif neighbour in self.liveCellCurrent:
                        # add to contiglist if it is not there
                        if neighbour not in self.contigList:
                            self.contigList.append(neighbour)
                            # call contiguous on this live neighbour to perform this same loop on each of its neighbours
                            self.contiguous(neighbour)
                        else:
                            next
                    else:
                        # neighbour is not alive
                        next

    def calc_dist(self, x, y, z):

        dist = math.sqrt(
            (x - self.starting_position) ** 2 + (y - self.starting_position) ** 2 + (z - self.starting_position) ** 2)
        return dist

    def calc_signature(self):
        # Ai defines the total number of living cells with i living neighbours
        # Dj gives the total number of dead cells in the vicinity of the config with j living
        #neighbours
        #D0 represents space and is omitted

        for cell in self.contigList:
            xcoord = cell[0]
            ycoord = cell[1]
            zcoord = cell[2]

            for x in range(-1, 2):
                for y in range(-1, 2):
                    for z in range(-1, 2):
                        neighbour = (xcoord + x, ycoord + y, zcoord + z)
                        if x == 0 and y == 0 and z == 0:
                            next
                        elif neighbour in self.contigList:
                            # the cell is a live cell and its neighbour count will be in live_vicinity\
                            if neighbour in self.live_vicinity:
                                self.live_vicinity[neighbour] = self.live_vicinity[neighbour] + 1
                            else:
                                # else add the tuple as a key
                                self.live_vicinity[neighbour] = 1
                        else:
                            if neighbour in self.dead_vicinity:
                                self.dead_vicinity[neighbour] = self.dead_vicinity[neighbour] + 1
                            else:
                                # else add the tuple as a key
                                self.dead_vicinity[neighbour] = 1

        # now that we have the live neighbour counts separated into live and dead cells, split into two vectors

        for key, val in self.live_vicinity.items():
            self.live_hist[val] += 1

        for key, val in self.dead_vicinity.items():
            if val == 0:
                next
            else:
                self.dead_hist[val] += 1

        self.live_hist.reverse()

        self.signature = self.live_hist + self.dead_hist

    def return_signature(self):
        return self.signature

    def reward_calc(self):
        # get average number of cells
        self.avg_pop = (sum(self.populationList))/(len(self.populationList))

        #low dist small
        if (self.avg_pop <= 10) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 15):
            # self.imm_Rew = 7
            # self.imm_Rew = 0.07
            self.imm_Rew = 1
        # med dist small
        elif (self.avg_pop <= 12) and (self.most_extreme_dist > 15) and (self.most_extreme_dist <= 20):
            # self.imm_Rew = 7
            # self.imm_Rew = 0.07
            self.imm_Rew = 2
        # long dist small, easy to make
        if (self.avg_pop <= 12) and (self.most_extreme_dist > 25):
            # self.imm_Rew = 14
            # self.imm_Rew = 0.14
            self.imm_Rew = 4

        # low dist medium or large
        elif (self.avg_pop > 12) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 15):
            # self.imm_Rew = 28
            # self.imm_Rew = 0.28
            self.imm_Rew = 3
        #   # low dist large, easy to make
        # elif (self.avg_pop > 22) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 15):
        #     # self.imm_Rew = 112
        #     # self.imm_Rew = 1.12
        #     self.imm_Rew = 3


        #med dist medium, harder to make
        elif (self.avg_pop <= 22) and (self.most_extreme_dist > 15) and (self.most_extreme_dist <= 20):
                # self.imm_Rew = 28
                # self.imm_Rew = 0.28
            self.imm_Rew = 38

        #med dist large, worth more than previous moves combined
        elif (self.avg_pop > 22) and (self.most_extreme_dist > 15) and (self.most_extreme_dist <= 20):
                # self.imm_Rew = 112
                # self.imm_Rew = 1.12
            self.imm_Rew = 38


        #long dist med, hardest to make
        elif (self.avg_pop <= 22) and (self.most_extreme_dist > 25):
            # self.imm_Rew = 56
            # self.imm_Rew = 0.56
            self.imm_Rew = 42

        #stable large
        elif (self.avg_pop > 22) and (self.most_extreme_dist > 25):
            #168
            # self.imm_Rew = 170
            # self.imm_Rew = 1.7
            self.imm_Rew = 50

    def check_if_oscillator(self):
        # if most extreme cell is not topped for 70 generations, we will assume that nothing neat was generated
        if (self.stopwatch > 0) and self.mightBeOsc:
            # check if the timer is greater than 50, indicating no progress is being made
            if self.stopwatch > 50:
                self.oscillator = True
            else:
                self.stopwatch += 1
        elif self.mightBeOsc:
            # start the timer
            self.stopwatch += 1
        else:
            self.stopwatch = 0

    # no need for timer, the most extreme cell is changing

    def getReward(self):
        return self.total_reward

    def getGliderV(self):
        return self.gliderV

    def resetSim(self):
        self.starting_position = 64

        self.numLiveCells = 0

        self.imm_Rew = 0
        self.avg_pop = 0

        self.generation = 1

        self.hitLastBorder = False
        self.stillLife = False
        self.allDead = False
        self.oscillator = False

        # list of tuples (indices of cells)
        self.liveCellCurrent = []
        self.liveCellPrev = []

        # keep track of live cells and their neighbours, if the number of neighbours is 5, 6, or 7
        # if a cell goes from these values to outside of this range check liveCellsCurrent and kick em
        # if a cell goes from outside these values to 6, add em
        self.changelist_n_counts = {}

        self.contigList = []

        self.nonContigPop = []
        self.populationList = []

        # MAKE BETTER VALUES

        self.rewardList = []
        self.current_reward = 0
        self.total_reward = 0
        self.most_extreme_dist = 0
        self.current_most_extreme_cell = (self.starting_position, self.starting_position, self.starting_position)
        self.mightBeOsc = False
        self.gliderV = False

        self.stopwatch = 0

        self.dead_vicinity = {}
        self.live_vicinity = {}
        self.dead_hist = []
        for i in range(1, 27):
            self.dead_hist.append(0)
        self.live_hist = []
        for i in range(0, 27):
            self.live_hist.append(0)
        self.signature = []

        # dist = 0
        # cells = 0
        #
        # #basic reward to get it going
        # if (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 15) and (self.avg_pop > 8) and (self.avg_pop <= 15):
        #     self.imm_Rew = 1
        #
        # #medium sized objects that go a medium distance
        # elif (self.most_extreme_dist > 15) and (self.most_extreme_dist <= 22) and (self.avg_pop > 8) and (self.avg_pop <= 23):
        #     dist = 2
        # elif (self.most_extreme_dist > 22) and (self.avg_pop > 8) and (self.avg_pop <= 15):
        #     dist = 3
        # elif (self.most_extreme_dist > 22) and (self.avg_pop > 15) and (self.avg_pop <= 23):
        #     dist = 3
        #
        # # elif (self.most_extreme_dist > 14) and (self.most_extreme_dist <= 18):
        # #     dist = 1.5
        # elif (self.most_extreme_dist > 15) and (self.most_extreme_dist <= 22):
        #     dist = 2
        # elif (self.most_extreme_dist > 22):
        #     dist = 3
        #
        #
        # if (self.avg_pop > 12) and (self.avg_pop <= 23):
        #     cells = 1.5
        # elif (self.avg_pop > 23):
        #     cells = 2.5
        #
        #
        # elif (self.avg_pop > 23) and (self.most_extreme_dist > 22):
        #      self.imm_Rew = 93
        # else:
        #     self.imm_Rew = cells + dist

        # the agent can now only receive 1 grade of reward once
        # it should not be able to farm by making simple patterns
        # it can combine rewards though so each grade must be more than those beneath it combined
        # the goal is to give it a base strategy that it can then branch off of
        # a general policy for drawing more stable objects

        # #low dist small
        # if (self.avg_pop <= 12) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 15):
        #     # self.imm_Rew = 7
        #     # self.imm_Rew = 0.07
        #     self.imm_Rew = 0.5
        # # low dist medium
        # elif (self.avg_pop <= 22) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 15):
        #     # self.imm_Rew = 28
        #     # self.imm_Rew = 0.28
        #     self.imm_Rew = 1
        # #
        #     # low dist large
        # elif (self.avg_pop > 22) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 15):
        #     # self.imm_Rew = 112
        #     # self.imm_Rew = 1.12
        #     self.imm_Rew = 2
        #
        # # med dist small, worth small dist medium
        # elif (self.avg_pop <= 12) and (self.most_extreme_dist > 15) and (self.most_extreme_dist <= 20):
        #         # self.imm_Rew = 7
        #         # self.imm_Rew = 0.07
        #     self.imm_Rew = 1
        # #med dist medium, worth small dist large
        # elif (self.avg_pop <= 22) and (self.most_extreme_dist > 15) and (self.most_extreme_dist <= 20):
        #         # self.imm_Rew = 28
        #         # self.imm_Rew = 0.28
        #     self.imm_Rew = 2
        # #med dist large, worth more than previous moves combined
        # elif (self.avg_pop > 22) and (self.most_extreme_dist > 15) and (self.most_extreme_dist <= 20):
        #         # self.imm_Rew = 112
        #         # self.imm_Rew = 1.12
        #     self.imm_Rew = 4
        #
        # #long dist small, worth med dist large
        # elif (self.avg_pop <= 12) and (self.most_extreme_dist > 25):
        #     # self.imm_Rew = 14
        #     # self.imm_Rew = 0.14
        #     self.imm_Rew = 4
        #
        # #long dist med, worth more than all combined
        # elif (self.avg_pop <= 22) and (self.most_extreme_dist > 25):
        #     # self.imm_Rew = 56
        #     # self.imm_Rew = 0.56
        #     self.imm_Rew = 9
        #
        # #stable large
        # elif (self.avg_pop > 22) and (self.most_extreme_dist > 25):
        #     #168
        #     # self.imm_Rew = 170
        #     # self.imm_Rew = 1.7
        #     self.imm_Rew = 18

        ##WELL THOUGGHT ONE
        # low dist small
        # if (self.avg_pop <= 12) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 15):
        #     # self.imm_Rew = 7
        #     # self.imm_Rew = 0.07
        #     self.imm_Rew = 0.5
        #     # med dist small, worth small dist medium
        # elif (self.avg_pop <= 12) and (self.most_extreme_dist > 15) and (self.most_extreme_dist <= 20):
        #     # self.imm_Rew = 7
        #     # self.imm_Rew = 0.07
        #     self.imm_Rew = 1
        # # long dist small, worth med dist large
        # elif (self.avg_pop <= 12) and (self.most_extreme_dist > 25):
        #     # self.imm_Rew = 14
        #     # self.imm_Rew = 0.14
        #     self.imm_Rew = 3
        #
        # # low dist medium
        # elif (self.avg_pop <= 22) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 15):
        #     # self.imm_Rew = 28
        #     # self.imm_Rew = 0.28
        #     self.imm_Rew = 2
        #     # med dist medium, worth small dist large
        # elif (self.avg_pop <= 22) and (self.most_extreme_dist > 15) and (self.most_extreme_dist <= 20):
        #     # self.imm_Rew = 28
        #     # self.imm_Rew = 0.28
        #     self.imm_Rew = 4
        #     # long dist med, worth more than all combined
        # elif (self.avg_pop <= 22) and (self.most_extreme_dist > 25):
        #     # self.imm_Rew = 56
        #     # self.imm_Rew = 0.56
        #     self.imm_Rew = 11
        #
        # #
        # # low dist large
        # elif (self.avg_pop > 22) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 15):
        #     # self.imm_Rew = 112
        #     # self.imm_Rew = 1.12
        #     self.imm_Rew = 5
        #
        # # med dist large, worth more than previous moves combined
        # elif (self.avg_pop > 22) and (self.most_extreme_dist > 15) and (self.most_extreme_dist <= 20):
        #     # self.imm_Rew = 112
        #     # self.imm_Rew = 1.12
        #     self.imm_Rew = 6
        #
        # # stable large
        # elif (self.avg_pop > 22) and (self.most_extreme_dist > 25):
        #     # 168
        #     # self.imm_Rew = 170
        #     # self.imm_Rew = 1.7
        #     self.imm_Rew = 34

        # # semi-stable small
        # if (self.avg_pop <= 12) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 25):
        #     # self.imm_Rew = 7
        #     # self.imm_Rew = 0.07
        #     self.imm_Rew = 1
        #
        #     # stable small
        # elif (self.avg_pop <= 12) and (self.most_extreme_dist > 25):
        #     # self.imm_Rew = 14
        #     # self.imm_Rew = 0.14
        #     self.imm_Rew = 2
        #
        # # semi-stable medium
        # elif (self.avg_pop <= 22) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 25):
        #     # self.imm_Rew = 28
        #     # self.imm_Rew = 0.28
        #     self.imm_Rew = 3
        #
        #     # semi-stable large, worth more than stable small to avoid local minimum
        # elif (self.avg_pop > 22) and (self.most_extreme_dist > 10) and (self.most_extreme_dist <= 25):
        #     # self.imm_Rew = 112
        #     # self.imm_Rew = 1.12
        #     self.imm_Rew = 4
        #
        # # stable medium
        # elif (self.avg_pop <= 22) and (self.most_extreme_dist > 25):
        #     # self.imm_Rew = 56
        #     # self.imm_Rew = 0.56
        #     self.imm_Rew = 11
        #
        # # stable large
        # elif (self.avg_pop > 22) and (self.most_extreme_dist > 25):
        #     # 168
        #     # self.imm_Rew = 170
        #     # self.imm_Rew = 1.7
        #     self.imm_Rew = 22