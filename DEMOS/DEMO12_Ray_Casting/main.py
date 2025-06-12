import numpy as np
import matplotlib.pyplot as plt
import trimesh
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_spherical_rays(ray_directions, origin, ray_length=1.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    origin = np.array(origin)  # ensure it's a NumPy array

    for direction in ray_directions:
        direction = np.array(direction) * ray_length
        end_point = origin + direction * ray_length
        ax.plot([origin[0], end_point[0]],
                [origin[1], end_point[1]],
                [origin[2], end_point[2]],
                color='blue')

    # Plot the origin point
    ax.scatter(*origin, color='red', s=50, label='Origin')

    # Adjust axis limits to center around origin
    max_extent = ray_length + np.max(np.abs(origin))
    ax.set_xlim([origin[0] - max_extent, origin[0] + max_extent])
    ax.set_ylim([origin[1] - max_extent, origin[1] + max_extent])
    ax.set_zlim([origin[2] - max_extent, origin[2] + max_extent])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Spherical Rays from Custom Origin')
    ax.legend()

    plt.show()


# Möller–Trumbore intersection
def ray_triangle_intersection(ray_orig, ray_dir, v0, v1, v2, epsilon=1e-6):
    # Step 1: Find vectors for two edges sharing v0
    e1 = v1 - v0
    e2 = v2 - v0

    # Step 2: Begin calculating determinant - also used to calculate u parameter
    p = np.cross(ray_dir, e2)  # p is perpendicular to ray_dir and edge2
    det = np.dot(e1, p)  # If det is close to 0, ray is parallel to triangle

    # Verifica se o raio é paralelo ao triângulo
    if abs(det) < epsilon:
        return None  # Sem interseção (paralelo)

    # distance from v0 to origin
    dist = ray_orig - v0  # T
    inv_det = 1.0 / det

    # Calculate u parameter and test bounds
    u = np.dot(dist, p) * inv_det
    if u < 0.0 or u > 1.0:
        return None  # Intersection lies outside the triangle

    # Calculate Q vector
    q = np.cross(dist, e1)

    # Calculate v parameter and test bounds
    v = np.dot(ray_dir, q) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return None  # Intersection lies outside the triangle

    # Compute t (distance from O to intersection point)
    t = np.dot(e2, q) * inv_det
    if t >= epsilon:
        # Ray intersection
        intersection_point = ray_orig + ray_dir * t
        return intersection_point
    else:
        return None  # Line intersects but not the ray


def generate_spherical_rays(theta_begin, theta_end, phi_begin, phi_end, num_rays):
    ray_directions = []
    for phi in np.linspace(phi_begin, phi_end, num_rays):
        for theta in np.linspace(theta_begin, theta_end, num_rays):
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            direction = np.array([x, y, z])
            direction = direction / np.linalg.norm(direction)
            ray_directions.append(direction)

    return ray_directions


def plot_mesh_with_rays_and_hits(mesh, ray_orig, ray_directions, intersections, hit_faces, ray_length=10.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot mesh faces
    mesh_faces = []
    face_colors = []

    for i, face in enumerate(mesh.faces):
        verts = mesh.vertices[face]
        mesh_faces.append(verts)
        if i in hit_faces:
            face_colors.append([1.0, 1.0, 0.0, 1.0])  # Yellow for hit
        else:
            face_colors.append([0.8, 0.8, 0.8, 0.2])  # Light gray otherwise

    mesh_collection = Poly3DCollection(mesh_faces, facecolors=face_colors, linewidths=0.1, edgecolors='k', alpha=0.9)
    ax.add_collection3d(mesh_collection)

    # Plot rays
    for direction in ray_directions:
        direction = np.array(direction) * ray_length
        end_point = ray_orig + direction * ray_length
        ax.plot([ray_orig[0], end_point[0]],
                [ray_orig[1], end_point[1]],
                [ray_orig[2], end_point[2]],
                color='yellow', alpha=0.3)

    # Plot ray origin
    ax.scatter(*ray_orig, color='red', s=50, label='Ray Origin')

    # Plot intersection points
    if intersections:
        inters = np.array(intersections)
        ax.scatter(inters[:, 0], inters[:, 1], inters[:, 2], color='black', s=5, label='Intersections')

    # Axis settings
    all_verts = mesh.vertices
    xyz_min = all_verts.min(axis=0)
    xyz_max = all_verts.max(axis=0)
    pad = 5.0

    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    ax.set_xlim([xyz_min[0] - pad, xyz_max[0] + pad])
    ax.set_ylim([xyz_min[1] - pad, xyz_max[1] + pad])
    ax.set_zlim([xyz_min[2] - pad, xyz_max[2] + pad])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Mesh with Rays and Intersections")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    mesh = trimesh.load('DEMOS/DEMO12_Ray_Casting/torus_paraview.obj')

    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(geometries)

    # Center the mesh
    mesh.apply_translation(-mesh.centroid)

    # Normalize and scale uniformly
    current_size = mesh.extents.max()
    mesh.apply_scale(4.0 / current_size)

    mesh._cache.clear()


    bbox_min, bbox_max = mesh.bounds
    centroid = mesh.centroid
    print("Bounding Box Min:", bbox_min)
    print("Bounding Box Max:", bbox_max)
    print("Centroid:", centroid)

    # Calculate ray origin position (directly above the torus)
    ray_height = bbox_max[2] + 2.0  # 2 units above the highest point
    ray_orig = np.array([centroid[0], centroid[1], ray_height])



    ray_directions = generate_spherical_rays(theta_begin=0,
                                             theta_end=2 * np.pi,
                                             phi_begin=0,
                                             phi_end=np.pi,
                                             num_rays=50)

    # plot_spherical_rays(origin=ray_orig, ray_directions=ray_directions)

    intersections = []
    hit_faces = []
    for ray_dir in tqdm(ray_directions, desc="Casting rays"):
        for face_index, face in enumerate(mesh.faces):
            v0 = np.array(mesh.vertices[face[0]])
            v1 = np.array(mesh.vertices[face[1]])
            v2 = np.array(mesh.vertices[face[2]])

            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal)
            if np.dot(normal, ray_dir) >= 0:
                continue  # Cull back-facing triangle

            inter_point = ray_triangle_intersection(ray_orig, ray_dir, v0, v1, v2, 1e-4)
            if inter_point is not None:
                intersections.append(inter_point)
                hit_faces.append(face_index)

    hit_faces = list(set(hit_faces))  # Remove duplicates

    print(f"Found {len(intersections)} intersections.")
    print(hit_faces)

    plot_mesh_with_rays_and_hits(mesh, ray_orig, ray_directions, intersections, hit_faces, ray_length=10.0)

