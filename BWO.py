import random
import time
import numpy as np
import random as rn


def getPheromone(fit, min, max):  # Eq.12 in the paper
    o = np.zeros((fit.shape[0]))
    for i in range(fit.shape[0]):
        o[i] = (max - fit[i]) / (max - min)
    return o


def getBinary():  # algorithm-2 in the paper page 6.
    if np.random.random() < 0.5:
        val = 0
    else:
        val = 1
    return val

# Black Widow Optimization (BWO)
def BWO(Positions, fobj, Lb, Ub, Max_iter):
    ### Black Widow Algorithm
    SearchAgents_no = Positions.shape[0]
    dim =  Positions.shape[1]
    ub = Ub[0, :]
    lb = Lb[1, :]
    Fitness = np.zeros((SearchAgents_no))
    for i in range(SearchAgents_no):
        Fitness[i] = fobj(Positions[i, :])

    [vMin, minIdx] = np.min(Fitness), np.argmin(Fitness)  # the min fitness value vMin and the position minIdx
    theBestVct = Positions[minIdx, :]  # the best vector
    [vMax, maxIdx] = np.max(Fitness), np.argmax(Fitness)   # the min fitness value vMin and the position minIdx
    Convergence_curve = np.zeros((Max_iter))
    Convergence_curve[0] = vMin
    pheromone = getPheromone(Fitness, vMin, vMax)
    ct = time.time()

    # Main loop
    for t in range(Max_iter):
        beta = -1 + 2 * np.random.random()  # -1 < beta2 < 1     section 3.2.1 in the paper, page 4
        m = 0.4 + 0.5 * np.random.random()  # 0.4 < m < 0.9
        v = np.zeros((SearchAgents_no, dim))
        for r in range(SearchAgents_no):
            P = np.random.random()
            r1 = round(1 + (SearchAgents_no - 2) * np.random.random())
            if P >= 0.3:  # spiral search   Eq. 11 in the paper, page 4
                v[r, :] = theBestVct - np.cos(2 * np.pi * beta) * Positions[r, :]
            else:  # direct search Eq. 11
                v[r, :] = theBestVct - m * Positions[r1, :]

            if pheromone[r] <= 0.3:
                band = 1
                while band:
                    r1 = round(1 + (SearchAgents_no - 2) * np.random.random())
                    r2 = round(1 + (SearchAgents_no - 2) * np.random.random())
                    if r1 != r2:
                        band = 0
                        # pheromone function.  Eq. 13 page 5 , getBinary is the  algorithm-2 in the paper, page 6
                v[r, :] = theBestVct + (Positions[r1, :] - ((-1) ^ getBinary()) * Positions[r2, :]) / 2

            # ********************************************************************************
            # Return back the search agents that go beyond the boundaries of the search space
            Flag4ub = v[r, :] > ub
            Flag4lb = v[r, :] < lb
            v[r, :] = (v[r, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
            # Evaluate new solutions
            Fnew = fobj(v[r, :])
            # Update if the solution improves
            if Fnew <= Fitness[r]:
                Positions[r, :] = v[r, :]
                Fitness[r] = Fnew

            if Fnew <= vMin:
                theBestVct = v[r, :]
                vMin = Fnew

        # update max and pheromons
        [vMax, maxIdx] = np.max(Fitness), np.argmax(Fitness)
        pheromone = getPheromone(Fitness, vMin, vMax)
        Convergence_curve[t] = vMin
    ct = time.time() - ct
    # **********************************[End  BWOA function]
    return vMin, Convergence_curve, theBestVct, ct
