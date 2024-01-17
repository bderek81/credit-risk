import time, random

num_layers = 200

def ReLU(x):
    return x if x > 0 else 0.5*x

def dReLU(x):
    return 1 if x > 0 else 0.5

def DNN(x, a, b, return_list):
    h = [x]
    for i in range(num_layers):
        h.append(ReLU(a[i]*h[i] + b[i]))
    
    return h if return_list else h[num_layers]

def L(x):
    return x*x

def BP(x, a, b):
    h = DNN(x, a, b, True)
    dLdh = [2*h[num_layers]]
    for i in range(num_layers-1, 0, -1):
        dLdh.insert(0, dLdh[0]*dReLU(a[i]*h[i] + b[i])*a[i])
    
    dLda = [
        dLdh[i] * dReLU(a[i]*h[i] + b[i])*h[i]
        for i in range(num_layers)
    ]
    dLdb = [
        dLdh[i] * dReLU(a[i]*h[i] + b[i])
        for i in range(num_layers)
    ]
    
    return dLda + dLdb

def FD(x, a, b):
    base = L(DNN(x, a, b, False))
    eps = 0.001
    dLda = [0 for i in range(num_layers)]
    dLdb = [0 for i in range(num_layers)]
    for i in range(num_layers):
        aeps = a.copy()
        aeps[i] += eps
        beps = b.copy()
        beps[i] += eps
        dLda[i] = (L(DNN(x, aeps, b, False)) - base) / eps
        dLdb[i] = (L(DNN(x, a, beps, False)) - base) / eps
    return dLda + dLdb

my_x = random.random() * 100
a = [random.random() * 100 for _ in range(num_layers)]
b = [random.random() * 100 for _ in range(num_layers)]

t1 = time.time()
for i in range(100):
    deriv = BP(my_x, a, b)

t2 = time.time()
for i in range(100):
    deriv = FD(my_x, a, b)

t3 = time.time()

print(f"avg time for backpropagation: {(t2-t1)/100}")
print(f"avg time for finite difference: {(t3-t2)/100}")