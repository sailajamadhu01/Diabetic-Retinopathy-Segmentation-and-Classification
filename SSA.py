import numpy as np
import time


def bounds(s, lb, ub):
    s = np.maximum(s, lb)
    s = np.minimum(s, ub)
    return s


# Example objective function (Rastrigin function)
def rastrigin(x):
    A = 10
    return A * len(x) + sum((x ** 2 - A * np.cos(2 * np.pi * x)))


# Sparrow Search Algorithm (SSA)
def SSA(x, fobj, lb, ub, M):
    pNum, dim = x.shape

    fit = np.apply_along_axis(fobj, 1, x)

    pFit = fit.copy()
    pX = x.copy()
    fMin = np.min(fit)
    bestI = np.argmin(fit)
    bestX = x[bestI, :]

    Convergence_curve = np.zeros(M)
    ct = time.time()

    # Start updating the solutions
    for t in range(M):
        sortIndex = np.argsort(pFit)
        fmax = np.max(pFit)
        B = np.argmax(pFit)
        worse = x[B, :]

        r2 = np.random.rand()
        if r2 < 0.8:
            for i in range(pNum):
                r1 = np.random.rand()
                x[sortIndex[i], :] = pX[sortIndex[i], :] * np.exp(-i / (r1 * M))
                x[sortIndex[i], :] = bounds(x[sortIndex[i], :], lb[sortIndex[i]], ub[sortIndex[i]])
                fit[sortIndex[i]] = fobj(x[sortIndex[i], :])
        else:
            for i in range(pNum):
                x[sortIndex[i], :] = pX[sortIndex[i], :] + np.random.randn(1) * np.ones(dim)
                x[sortIndex[i], :] = bounds(x[sortIndex[i], :], lb[sortIndex[i]], ub[sortIndex[i]])
                fit[sortIndex[i]] = fobj(x[sortIndex[i], :])

        bestII = np.argmin(fit)
        bestXX = x[bestII, :]

        for i in range(pNum):
            A = (np.floor(np.random.rand(dim) * 2) * 2 - 1)
            if i > (pNum / 2):
                x[sortIndex[i], :] = np.random.randn(1) * np.exp((worse - pX[sortIndex[i], :]) / (i ** 2))
            else:
                x[sortIndex[i], :] = bestXX + (np.abs((pX[sortIndex[i], :] - bestXX))) * np.dot(A.reshape(-1, 1),
                                                                                                np.linalg.pinv(
                                                                                                    A.reshape(-1, 1)))[
                                                                                         :, 0] * np.ones(dim)
            x[sortIndex[i], :] = bounds(x[sortIndex[i], :], lb[sortIndex[i]], ub[sortIndex[i]])
            fit[sortIndex[i]] = fobj(x[sortIndex[i], :])

        c = np.random.permutation(sortIndex)
        b = sortIndex[c[:20]]

        for j in b:
            if pFit[sortIndex[j]] > fMin:
                x[sortIndex[j], :] = bestX + (np.random.randn(1, dim) * (np.abs(pX[sortIndex[j], :] - bestX)))
            else:
                x[sortIndex[j], :] = pX[sortIndex[j], :] + (2 * np.random.rand() - 1) * (
                    np.abs(pX[sortIndex[j], :] - worse)) / (pFit[sortIndex[j]] - fmax + 1e-50)
            x[sortIndex[j], :] = bounds(x[sortIndex[j], :], lb[sortIndex[j]], ub[sortIndex[j]])
            fit[sortIndex[j]] = fobj(x[sortIndex[j], :])

        for i in range(pNum):
            if fit[i] < pFit[i]:
                pFit[i] = fit[i]
                pX[i, :] = x[i, :]

            if pFit[i] < fMin:
                fMin = pFit[i]
                bestX = pX[i, :]

        Convergence_curve[t] = fMin
    ct = ct - time.time()
    return fMin, Convergence_curve, bestX, ct

