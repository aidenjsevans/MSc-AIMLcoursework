from argminsum import *

def minsumcomb(x, M):
    N = len(x)
    P = [[zero] * (M + 1) for _ in range(N + 1)]

    for m in range(1, M + 1):
        P[0][m] = inf

    for n in range(1, N + 1):
        for m in range(1, M + 1):
            P[n][m] = min(P[n-1][m], P[n-1][m-1].__add__(argminsum(x[n-1], [x[n-1]])))

    return P

def main():
    M = 2
    arr = [5,-2,3,7,0]
    print(f"\nInput array: {arr}")

    P = minsumcomb(arr, M)
    print(f"\nRecursive process: \n")

    for row in P:
        print(f"\t{row}")

    Popt = P[-1][-1]

    print(f"\nOptimal configuration of {arr} of size {M}: {Popt}")

if __name__ == '__main__':
    main()
