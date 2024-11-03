import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Cube data
cube = np.array([
    [[1, 47, 28, 54, 54],
     [20, 9, 62, 39, 11],
     [64, 18, 37, 11, 26],
     [45, 56, 3, 26, 39],
     [5, 27, 52, 46, 50]],

    [[25, 7, 52, 46, 5],
     [60, 17, 38, 25, 40],
     [40, 58, 13, 19, 15],
     [19, 15, 50, 46, 3],
     [60, 45, 32, 53, 12]],

    [[35, 61, 10, 24, 33],
     [22, 33, 12, 35, 1],
     [2, 43, 32, 30, 41],
     [30, 4, 55, 41, 41],
     [53, 12, 24, 16, 16]],

    [[6, 41, 8, 29, 16],
     [42, 51, 8, 29, 57],
     [43, 14, 57, 36, 36],
     [34, 31, 34, 16, 29],
     [52, 21, 34, 16, 36]],

    [[59, 16, 31, 42, 34],
     [50, 24, 21, 31, 46],
     [51, 41, 37, 49, 29],
     [52, 18, 21, 16, 34],
     [51, 16, 30, 42, 29]]
])

fig = plt.figure(figsize=(20, 50))
ax = fig.add_subplot(111, projection='3d')

n = cube.shape[0]
cube_size = 1.0
layer_spacing = 120.0  
cube_height = 40.0 

colors =['#ffcccc', '#cce5ff', '#ccffcc', '#ffe6cc', '#e6e6fa']

for z in range(n):
    for y in range(n):
        for x in range(n):

            x_pos, y_pos, z_pos = x, y, z * layer_spacing + cube_height
            face_color = colors[z]

            ax.bar3d(x_pos, y_pos, z_pos, cube_size, cube_size, cube_height, color=face_color, alpha=0.15)
            
            ax.text(x_pos + cube_size / 2, y_pos + cube_size / 2, z_pos + cube_height/2, 
                    str(cube[z, y, x]), ha='center', va='center', color="black", fontsize=6, weight='bold', 
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Layer (Z)')
ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(-0.5, n - 0.5)
ax.set_zlim(-0.5, n * layer_spacing + cube_height)


ax.view_init(elev=20, azim=30)

plt.title("3D Visualization of The 5x5x5 Magic Cube")
plt.show()
