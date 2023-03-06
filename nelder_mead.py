import copy


def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):

    # Init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)

        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    iters = 0
    while 1:
        # Order
        res.sort(key = lambda x: x[1])
        best = res[0][1]

        # break the while loop after max_iter has been reached.
        if max_iter and iters >= max_iter:
            return res[0]
        iters +=1

        # break while after no_improv_break with no improvements has been reached
        print(f'...best so far: {best}')

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]
        
        # Calculate the centroid point. I.e average all points except last.
        x0 = [0]*dim
        for i in res[:-1]:
            for j,k in enumerate(i[0]):
                x0[j] += k/(len(res)-1)
        
        # Reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <- rscore <= res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # Expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(res[-1][0] - x0)
            escore = f(xe)

            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue
     
        # Contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # Shrink
        x1 = res[0][0]
        new_res = []
        for i in res:
            shrink = x1 + sigma*(i[0] - x1)
            score = f(shrink)
            new_res.append([shrink, score])
        res = new_res


if __name__ == "__main__":
    import math
    import numpy as np

    def f(x, sigma=1, mu=1):
        return (2*math.pi*sigma**2)**(-len(x)/2)*math.exp((-1/(2*sigma**2))*sum((x - mu)**2))
    
    print(nelder_mead(f, np.array([0., 0., 0.])))