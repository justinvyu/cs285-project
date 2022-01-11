import pickle as pkl
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--buffer', type=str, required=True)
parser.add_argument('--buffer_key', type=str, default='observations')
args = parser.parse_args()

tick_size = 0.1

with open(args.buffer, 'rb') as f:
    data = pkl.load(f) 
    buffer_data = data[f'replay_buffer/{args.buffer_key}']

data_x = []
data_y = []

for g in buffer_data['observation']:
    data_x.append(g[0])
    data_y.append(g[1])    

 
heatmap, xedges, yedges = np.histogram2d(data_x, data_y, bins=50)
plt.hist(heatmap.flatten(), bins=200)
plt.savefig('distribution.png')
plt.clf()
heatmap = np.rot90(heatmap)

plt.pcolormesh(xedges, yedges, heatmap, cmap='OrRd')
plt.savefig('heatmap.png')

