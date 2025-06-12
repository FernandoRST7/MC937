#include <iostream>
#include <vector>
#include <array>
#include<glm/glm.hpp>
#include <memory>

// Axis-Aligned Bounding Box (AABB) structure
struct AABB
{
    glm::vec3 min_corner{}; // Minimum corner of the bounding box
    glm::vec3 max_corner{}; // Maximum corner of the bounding box

    AABB() {}

    // Constructor that initializes the bounding box with given min and max
    AABB(const glm::vec3& min, const glm::vec3& max):
        max_corner{max},
        min_corner{min}
    {}

    // Returns the axis (0=x, 1=y, 2=z) where the box is largest
    unsigned short getLargestAxis() const
    {
        glm::vec3 size = max_corner - min_corner;

        if (size.x >= size.y && size.x >= size.z)
        {
            return 0;
        }
        else if (size.y >= size.z)
        {
            return 1;
        }
        else
        {
            return 2;
        }
    }

    // Allows printing the AABB to the output
    friend std::ostream& operator<<(std::ostream& os, const AABB& aabb) {
        os << "AABB Corners:\n"
           << "  Min: (" << aabb.min_corner.x << ", "
           << aabb.min_corner.y << ", "
           << aabb.min_corner.z << ")\n"
           << "  Max: (" << aabb.max_corner.x << ", "
           << aabb.max_corner.y << ", "
           << aabb.max_corner.z << ")";
        return os;
    }

    // Splits the AABB into two along a specified axis
    std::pair<AABB, AABB> split(unsigned axis) const;
};

// Splits the AABB into two halves along the given axis
std::pair<AABB, AABB> AABB::split(unsigned axis) const
{
    float mid_point = (min_corner[axis] + max_corner[axis]) * 0.5f;
    std::cout << mid_point << "mid \n";

    glm::vec3 first_max{max_corner};
    first_max[axis] = mid_point;

    glm::vec3 second_min{min_corner};
    second_min[axis] = mid_point;

    return {
        AABB(min_corner, first_max),
        AABB(second_min, max_corner)
    };
}

// Structure representing a 3D mesh
struct Mesh
{
    using coordinate_t = std::vector<glm::vec3>; // List of vertex positions
    using triangles_t = std::vector<std::array<unsigned, 3>>; // List of triangle indices

    AABB aabb; // Bounding box for the mesh
    coordinate_t coordinates; // Vertex positions
    triangles_t triangles; // Triangles made of 3 vertex indices

    // Constructor that takes coordinates and triangle data
    Mesh(const coordinate_t& coordinates,
         const triangles_t& triangles):
        coordinates{coordinates},
        triangles{triangles}
    {
        updateAABB(); // Automatically compute the AABB
    }

    void updateAABB(); // Updates the mesh's bounding box

    // Sorts triangles based on their centroid's position along a given axis
    void sortTrianglesByAxis(unsigned axis)
    {
        auto comparator = [&](const std::array<unsigned, 3>& a, const std::array<unsigned, 3>& b)
        {
            glm::vec3 centroid_a = (coordinates[a[0]] + coordinates[a[1]] + coordinates[a[2]]) / 3.0f;
            glm::vec3 centroid_b = (coordinates[b[0]] + coordinates[b[1]] + coordinates[b[2]]) / 3.0f;
            return centroid_a[axis] < centroid_b[axis];
        };

        std::sort(triangles.begin(), triangles.end(), comparator);
    }

    // Splits the mesh's triangle list into two halves
    std::pair<Mesh, Mesh> splitMesh();
};

// Splits the mesh into two submeshes based on the triangle list
std::pair<Mesh, Mesh> Mesh::splitMesh()
{
    const size_t split_index = triangles.size() / 2;

    triangles_t first_t, second_t;

    // Divide triangles into two halves
    first_t.assign(triangles.begin(), triangles.begin() + split_index);
    second_t.assign(triangles.begin() + split_index, triangles.end());

    return {{coordinates, first_t}, {coordinates, second_t}};
}

// Computes the AABB that encloses all the mesh's vertices
void Mesh::updateAABB()
{
    glm::vec3 min{coordinates[0]};
    glm::vec3 max{coordinates[0]};

    for(const auto& coord: coordinates)
    {
        min = glm::min(min, coord);
        max = glm::max(max, coord);
    }

    aabb = {min, max};
}

// Node in the AABB tree
struct AABBNode
{
    Mesh mesh;
    bool leaf{false}; // True if this node is a leaf (only one triangle)

    std::unique_ptr<AABBNode> left_child{nullptr};
    std::unique_ptr<AABBNode> right_child{nullptr};

    AABBNode(const Mesh& mesh):mesh{mesh}{};

    bool isLeaf() const {return leaf;}
    void makeLeaf() { leaf = true;}
};

// AABB Tree structure
struct AABBTree
{
    std::unique_ptr<AABBNode> root{nullptr}; // Root node of the tree

    AABBTree(const Mesh& mesh); // Constructor builds root node
    void build();               // Starts recursive construction
    void print(std::unique_ptr<AABBNode>& node) const; // Prints tree content

private:
    void build(std::unique_ptr<AABBNode>& node); // Recursive builder
};

// Prints the first triangle of each leaf node in the tree
void AABBTree::print(std::unique_ptr<AABBNode>& node) const
{
    if(node==nullptr) {return;}
    if(!node->isLeaf())
    {
        print(node->left_child);
        print(node->right_child);
    }

    auto triangle = node->mesh.triangles[0];
    for(auto t:triangle)
    {
        std::cout << node->mesh.coordinates[t][0] << ", ";
        std::cout << node->mesh.coordinates[t][1] << ", ";
        std::cout << node->mesh.coordinates[t][2] << "\n";
    }
    std::cout << '\n';
}

// Creates the root node of the AABB tree
AABBTree::AABBTree(const Mesh& mesh)
{
    root = std::make_unique<AABBNode>(mesh);
}

// Builds the AABB tree and prints its structure
void AABBTree::build()
{
    build(root);
    print(root);
    std::cout << "[ OK ] Build AABB tree\n";
}

// Recursively builds the AABB tree
void AABBTree::build(std::unique_ptr<AABBNode>& node)
{
    if(node == nullptr) { return; }
    if(node->isLeaf() ) { return; }

    // Stop recursion if there is only one triangle
    if(node->mesh.triangles.size() == 1)
    {
        node->makeLeaf();
        return;
    }

    // Determine the longest axis of the current AABB
    unsigned short largests_axis = node->mesh.aabb.getLargestAxis();

    // Sort triangles along that axis
    node->mesh.sortTrianglesByAxis(largests_axis);

    // Split the mesh into two
    auto sub_meshes = node->mesh.splitMesh();

    // Create child nodes
    node->left_child = std::make_unique<AABBNode>(sub_meshes.first);
    node->right_child = std::make_unique<AABBNode>(sub_meshes.second);

    // Recurse
    build(node->left_child);
    build(node->right_child);
}

// Main function to test the AABB tree with a cube mesh
int main()
{
    // Define 8 vertices of a cube
    std::vector<glm::vec3> coordinates = {
        {-1, -1, -1}, // 0
        { 1, -1, -1}, // 1
        { 1,  1, -1}, // 2
        {-1,  1, -1}, // 3
        {-1, -1,  1}, // 4
        { 1, -1,  1}, // 5
        { 1,  1,  1}, // 6
        {-1,  1,  1}  // 7
    };

    // Define triangles using the vertex indices
    std::vector<std::array<unsigned, 3>> triangles = {
        {4, 5, 6},  {4, 6, 7},  // front
        {1, 0, 3},  {1, 3, 2},  // back
        {0, 4, 7},  {0, 7, 3},  // left
        {5, 1, 2},  {5, 2, 6},  // right
        {0, 1, 5},  {0, 5, 4},  // bottom
        {3, 7, 6},  {3, 6, 2}   // top
    };

    // Create a mesh from the cube's geometry
    Mesh mesh{std::move(coordinates), std::move(triangles)};

    // Build the AABB tree from the mesh
    AABBTree aabb_tree{mesh};
    aabb_tree.build();

    return 0;
}
