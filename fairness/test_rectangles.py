import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# Data
xdata = [np.random.uniform(0.0,  1.0, 10), np.random.uniform(0.7,  1.0, 10)]
ydata = [np.random.uniform(0.0,  1.0, 10), np.random.uniform(0.0,  1.0, 10)]

# mean_x, mean_y = np.mean(x), np.mean(y)
# std_x, std_y = np.std(x), np.std(y)

# Plotting
fig, ax = plt.subplots(1)
boxes = [Rectangle((np.mean(x) - np.std(x), np.mean(y) - np.std(y)), 2 * np.std(x), 2 * np.std(y)) for x, y in zip(xdata, ydata)]
pc = PatchCollection(boxes, facecolor = ["#9b59b6", "#3498db"], edgecolor = ["#9b59b6", "#3498db"], linewidths=(2,), alpha = 0.5)
ax.add_collection(pc)
plt.show()