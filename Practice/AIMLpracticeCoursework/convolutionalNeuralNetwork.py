from ad import *

def convnet(x):

    #   Define the ReLU activation function

    def reluActivation(dualTerm: dual) -> dual:
        val = max(0, dualTerm.val)
        grad = dualTerm.grad * int(dualTerm.val >= 0)
        return dual(val, grad)

    #   Input values initialised as type dual
    #   The derivative of w1 with respect to w1 is 1, that is, dw1/dw1 = 1
    #   Other values have gradient of 0 with respect to w1

    x1 = dual(x[0], 0)
    x2 = dual(x[1], 0)
    x3 = dual(x[2], 0)
    x4 = dual(x[3], 0)
    x5 = dual(x[4], 0)

    #Initialise weights

    w = [1.2,-0.2]
    v = [-0.3,0.6,1.3,-1.5]

    w1 = dual(w[0],1)
    w2 = dual(w[1],0)

    v1 = dual(v[0],0)
    v2 = dual(v[1],0)
    v3 = dual(v[2],0)
    v4 = dual(v[3],0)

    #Calculate the hidden nodes

    x1w1 = x1.__mul__(w1)
    x2w2 = x2.__mul__(w2)

    x2w1 = x2.__mul__(w1)
    x3w2 = x3.__mul__(w2)

    x3w1 = x3.__mul__(w1)
    x4w2 = x4.__mul__(w2)

    x4w1 = x4.__mul__(w1)
    x5w2 = x5.__mul__(w2)

    print(f"\nx1w1={x1w1}")
    print(f"x2w2={x2w2}")
    print(f"x2w1={x2w1}")
    print(f"x3w2={x3w2}")
    print(f"x3w1={x3w1}")
    print(f"x4w2={x4w2}")
    print(f"x4w1={x4w1}")
    print(f"x5w2={x5w2}")

    #With ReLU activation functions

    z1 = reluActivation(x1w1.__add__(x2w2))
    z2 = reluActivation(x2w1.__add__(x3w2))
    z3 = reluActivation(x3w1.__add__(x4w2))
    z4 = reluActivation(x4w1.__add__(x5w2))

    z = [z1,z2,z3,z4]

    print(f"\nz1={z1}")
    print(f"z2={z2}")
    print(f"z3={z3}")
    print(f"z4={z4}")

    #Calculate output value

    z1v1 = z1.__mul__(v1)
    z2v2 = z2.__mul__(v2)
    z3v3 = z3.__mul__(v3)
    z4v4 = z4.__mul__(v4)

    print(f"\nz1v1={z1v1}")
    print(f"z2v2={z2v2}")
    print(f"z3v3={z3v3}")
    print(f"z4v4={z4v4}")

    y = reluActivation(z1v1.__add__(z2v2.__add__(z3v3.__add__(z4v4))))

    print(f"\ny={y}")

    return y,z

def main():

    x = [0.3, -1.5, 0.7, 2.1, 0.1]
    y,z = convnet(x)

if __name__ == '__main__':
    main()















