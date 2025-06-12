import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal


def plot(model_1, model_2, colors):
    """
    Visualizes two 3D models side by side with different coloring
    Args:
        model_1: First 3D model (coordinates only)
        model_2: Second 3D model (same coordinates but with lighting colors)
        colors: Per-vertex colors calculated from lighting model
    """
    # Create figure with two 3D subplots
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   subplot_kw={'projection': '3d'},
                                   figsize=(15, 8))

    # Plot first model (basic, unlit)
    ax1.scatter(model_1[:, 0], model_1[:, 2], model_1[:, 1], c='b', label='Model 1')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')  # Note: Z and Y are swapped for better visualization
    ax1.set_zlabel('Z')

    # Plot second model with Phong shading colors
    ax2.scatter(model_2[:, 0], model_2[:, 2], model_2[:, 1], c=colors[:, :3], label='Model 2')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.show()


if __name__ == '__main__':
    # Load 3D model data from CSV file
    file_name = 'bunny_normals.csv'
    model = np.loadtxt(file_name, delimiter=',')

    # Split into normals (first 3 columns) and coordinates (last 3 columns)
    normals = model[:, 0:3]
    coords = model[:, 3:]

    # Print shape to verify data loading
    print(model.shape)

    # Gold material properties for Phong model:
    # [ambientRGB, diffuseRGB, specularRGB] each with alpha=1.0
    material = np.array([
        [0.24725, 0.1995, 0.0745, 1.0],  # Ambient reflectivity
        [0.75164, 0.60648, 0.22648, 1.0],  # Diffuse reflectivity
        [0.628281, 0.555802, 0.366065, 1.0]  # Specular reflectivity
    ])

    # White light properties:
    # [ambientRGB, diffuseRGB, specularRGB] each with alpha=1.0
    lightColor = np.array([
        [1.0, 1.0, 1.0, 1.0],  # Ambient light
        [1.0, 1.0, 1.0, 1.0],  # Diffuse light
        [1.0, 1.0, 1.0, 1.0]  # Specular light
    ])

    # Material shininess exponent (controls specular highlight size)
    shininess = 51.2

    # Lighting calculation for each vertex
    colors = []
    light_pos = np.array([-10, 10, -10])  # Light position in 3D space

    for i in range(len(coords)):
        # Normalize normal vector
        N = normals[i] / np.linalg.norm(normals[i])

        # Calculate light direction vector (L)
        L = light_pos - coords[i]
        L = L / np.linalg.norm(L)

        # Calculate view direction vector (V) - toward camera at origin
        V = -coords[i] / np.linalg.norm(-coords[i])

        # Calculate reflection vector (R)
        R = -L - 2 * np.inner(N, -L) * N

        # Phong lighting components:
        # Ambient = light ambient * material ambient
        ambient = lightColor[0] * material[0]

        # Diffuse = light diffuse * material diffuse * max(N·L, 0)
        diffuse = (lightColor[1] * material[1]) * np.maximum(np.inner(N, L), 0.0)

        # Specular = light specular * material specular * (max(R·V, 0)^shininess)
        specular = (lightColor[2] * material[2]) * (np.maximum(np.inner(R, V), 0.0) ** shininess)

        # Combine components and clamp to [0,1] range
        final_color = np.clip(ambient + diffuse + specular, 0.0, 1.0)

        colors.append(final_color)

    # Convert color list to numpy array for matplotlib
    colors = np.array(colors)

    # Visualize original vs lit model
    plot(coords, coords, colors)