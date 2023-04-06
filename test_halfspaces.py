import numpy as np
import matplotlib.pyplot as plt

def wrap(theta, lower, upper):
    return (theta - lower) % (upper - lower) + lower

# p1 = np.array([-0.25, -0.25])
# p2 = np.array([0.5, 0.5])
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0, 2.0])

fig, axs = plt.subplots()
axs.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')


dx = p2[0] - p1[0]
dy = p2[1] - p1[1]#*np.sign(dx)
th = np.arctan2(dy, dx)
th_ = wrap(th, 0, np.pi)
print("dx: ", dx, "   dy: ", dy)
print("th: ", th, "   th_: ", th_)
eps = 0.2



### lower y
deps = eps/np.cos(th_)

a1 = np.array([dy/(dx), -1.])
bnd = deps if dy < 0 else -deps
b1 = a1[0]*p1[0] + a1[1]*p1[1] + bnd

x = np.linspace(-1, 1, 100)
y = (b1 - a1[0]*x)/a1[1]

axs.plot(x, y, 'g-')

### higher y
a2 =  np.array([dy/(dx), -1.])
bnd = -deps if dy < 0 else deps
b2 =  a2[0]*p2[0] + a2[1]*p2[1] + bnd

x = np.linspace(-1, 1, 100)
y = (b2 - a2[0]*x)/a2[1]

axs.plot(x, y, 'r-')




### lower x
deps = eps/np.sin(th_)
print("deps: ", deps)

a3 = np.array([-1/a1[0], -1.])
bnd = -deps if dy < 0 else deps
b3 = a3[0]*p1[0] + a3[1]*p1[1] + bnd

x = np.linspace(-1, 1, 100)
y = (b3 - a3[0]*x)/a3[1]

axs.plot(x, y, 'b--')

### higher x
a4 =  np.array([-1/a2[0], -1.])
bnd = deps if dy < 0 else -deps
b4 = a4[0]*p2[0] + a4[1]*p2[1] + bnd

x = np.linspace(-1, 1, 100)
y = (b4 - a4[0]*x)/a4[1]

axs.plot(x, y, 'c--')




# evaluate a point that is surely inside the polytope
x = 0.5*(p1 + p2)
if dx > 0:
    print(-a1.dot(x) <= -b1)
    print(a2.dot(x) <= b2)
else:
    print(a1.dot(x) <= b1)
    print(-a2.dot(x) <= -b2)

if dy > 0:
    print(a3.dot(x) <= b3)
    print(-a4.dot(x) <= -b4)
else:
    print(-a3.dot(x) <= -b3)
    print(a4.dot(x) <= b4)

axs.set_aspect('equal')
axs.grid(True)
plt.show()

