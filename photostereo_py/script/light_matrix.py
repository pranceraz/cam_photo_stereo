import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setup parameters
num_lights = 12
radius = 35.0  # mm, radius of circular light arrangement
height = 50.0  # mm, z-position of LEDs
origin = np.array([0.0, 0.0, 15.0])  # object center (on optical axis)

# Compute light positions
light_positions = []
light_matrix = []

for i in range(num_lights):
    angle = 2 * np.pi * i / num_lights
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = height
    pos = np.array([x, y, z])
    light_positions.append(pos)

    direction = origin - pos  # direction: object → light
    unit_vector = direction / np.linalg.norm(direction)
    light_matrix.append(unit_vector)

light_positions = np.array(light_positions)
light_matrix = np.array(light_matrix)

# Print light matrix
print("Light matrix (object → light directions):")
print(np.round(light_matrix, 4))

# 3D Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot light positions
ax.scatter(light_positions[:, 0], light_positions[:, 1], light_positions[:, 2], c='blue', label='Light Sources')
ax.scatter(origin[0], origin[1], origin[2], c='red', label='Object Origin')

# Plot direction vectors from object to each light
for pos in light_positions:
    vec = pos - origin
    unit_vec = vec / np.linalg.norm(vec)
    ax.quiver(origin[0], origin[1], origin[2],
              unit_vec[0], unit_vec[1], unit_vec[2],
              length=np.linalg.norm(vec), color='green', arrow_length_ratio=0.1)


# Aesthetics
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Z [mm]")
ax.set_title("Light Directions (Object → LED)")
ax.legend()
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.show()
