import os
import matplotlib.pyplot as plt
import numpy as np

from sampler import Sampler

z_dim = 12
x_dim = 1080
y_dim = 1080

scale = 10
zscale = 0.001
n_image = 2000
window = 25
output_dir = 'results_large'
os.makedirs(output_dir, exist_ok=True)

sampler = Sampler(z_dim=z_dim, scale=scale)
wt1 = 1
wt2 = 0
z1 = np.random.uniform(-1, 1, size=(1, z_dim))
z2 = np.random.uniform(-1, 1, size=(1, z_dim))
z_shift = z1 * scale * zscale
z = np.copy(z1)

for i in range(n_image):
    z += z_shift
    x = sampler.generate(z, x_dim=x_dim, y_dim=y_dim)
    
    impath = os.path.join(output_dir, 'im' + str(i) + '.png') 
    plt.imshow(x, cmap='gray')
    plt.axis('off')
    plt.savefig(impath, bbox_inches='tight')

    wt1 -= 1 / window
    wt2 += 1 / window
    z_shift = (wt1 * z1 + wt2 * z2) * scale * zscale

    if i % 25 == 0:
        wt1 = 1
        wt2 = 0
        z1 = np.copy(z2)
        z2 = np.random.uniform(-1, 1, size=(1, z_dim))
        z_shift = (wt1 * z1 + wt2 * z2) * scale * zscale
