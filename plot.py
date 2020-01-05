from matplotlib import pyplot as plt
import numpy as np
x = np.linspace(0, 2 * np.pi, 40)
y=np.random.randint(0,50,size = 40)
plt.plot(x,y)
plt.show()