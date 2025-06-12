#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <fstream>
#include <string>

#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <random>

#include <optional>

struct ModelData {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
};

ModelData read_obj(const char* input_filename){
    /*Opens an obj file, saves the vectors, faces and normals, creates the verticces vector and returns it*/
    // Open the file
std::ifstream file(input_filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << input_filename << std::endl;
        return {};
    }

    // Dados temporários
    std::vector<glm::vec3> temp_vertices;
    std::vector<glm::vec3> temp_normals;
    std::vector<glm::ivec3> faces_vertex_indices;
    std::vector<glm::ivec3> faces_normal_indices;  // Novo: índices de normais das faces

    bool has_predefined_normals = false;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {  // Vértices
            float x, y, z;
            iss >> x >> y >> z;
            temp_vertices.push_back(glm::vec3(x, y, z));
        }
        else if (prefix == "vn") {  // Normais pré-definidas
            float nx, ny, nz;
            iss >> nx >> ny >> nz;
            temp_normals.push_back(glm::vec3(nx, ny, nz));
            has_predefined_normals = true;
        }
        else if (prefix == "f") {  // Faces
            std::string v1, v2, v3;
            iss >> v1 >> v2 >> v3;

            auto parse_face_token = [](const std::string& token) -> std::pair<int, int> {
                std::istringstream token_stream(token);
                std::string part;
                int vertex_idx = -1, normal_idx = -1;

                std::getline(token_stream, part, '/');
                if (!part.empty()) vertex_idx = std::stoi(part);

                // Pula a coordenada de textura (se existir)
                if (std::getline(token_stream, part, '/')) {
                    if (std::getline(token_stream, part, '/') && !part.empty()) {
                        normal_idx = std::stoi(part);
                    }
                }

                return {vertex_idx, normal_idx};
            };

            auto [v_idx1, n_idx1] = parse_face_token(v1);
            auto [v_idx2, n_idx2] = parse_face_token(v2);
            auto [v_idx3, n_idx3] = parse_face_token(v3);

            faces_vertex_indices.emplace_back(v_idx1, v_idx2, v_idx3);
            
            if (has_predefined_normals) { // NÃO SEI SE DEVERIA SE ISSO OU NOT ISSO
                faces_normal_indices.emplace_back(n_idx1, n_idx2, n_idx3);
            }
        }
    }

    // Processamento das normais
    ModelData model_data;

    if (has_predefined_normals && !temp_normals.empty()) {
        // Caso 1: Usa normais pré-definidas do arquivo
        for (size_t i = 0; i < faces_vertex_indices.size(); ++i) {
            const auto& v_indices = faces_vertex_indices[i];
            const auto& n_indices = faces_normal_indices[i];

            model_data.vertices.push_back(temp_vertices[v_indices.x - 1]);
            model_data.vertices.push_back(temp_vertices[v_indices.y - 1]);
            model_data.vertices.push_back(temp_vertices[v_indices.z - 1]);

            model_data.normals.push_back(temp_normals[n_indices.x - 1]);
            model_data.normals.push_back(temp_normals[n_indices.y - 1]);
            model_data.normals.push_back(temp_normals[n_indices.z - 1]);
        }
    } else {
        // Caso 2: Calcula normais a partir da geometria (como na versão anterior)
        std::vector<glm::vec3> vertex_normals(temp_vertices.size(), glm::vec3(0.0f));
        std::vector<int> vertex_counts(temp_vertices.size(), 0);

        for (const auto& face : faces_vertex_indices) {
            const int idx0 = face.x - 1;
            const int idx1 = face.y - 1;
            const int idx2 = face.z - 1;

            const glm::vec3& v0 = temp_vertices[idx0];
            const glm::vec3& v1 = temp_vertices[idx1];
            const glm::vec3& v2 = temp_vertices[idx2];

            glm::vec3 normal = glm::cross(v1 - v0, v2 - v0);
            float length = glm::length(normal);
            
            normal = glm::normalize(normal);
            vertex_normals[idx0] += normal;
            vertex_normals[idx1] += normal;
            vertex_normals[idx2] += normal;
            vertex_counts[idx0]++;
            vertex_counts[idx1]++;
            vertex_counts[idx2]++;
            
        }

        // Normalização final
        for (size_t i = 0; i < vertex_normals.size(); ++i) {
            if (vertex_counts[i] > 0) {
                vertex_normals[i] = glm::normalize(vertex_normals[i]);
            }
        }

        // Constrói o modelo de saída
        for (const auto& face : faces_vertex_indices) {
            const int idx0 = face.x - 1;
            const int idx1 = face.y - 1;
            const int idx2 = face.z - 1;

            model_data.vertices.push_back(temp_vertices[idx0]);
            model_data.vertices.push_back(temp_vertices[idx1]);
            model_data.vertices.push_back(temp_vertices[idx2]);

            model_data.normals.push_back(vertex_normals[idx0]);
            model_data.normals.push_back(vertex_normals[idx1]);
            model_data.normals.push_back(vertex_normals[idx2]);
        }
    }

    return model_data;
}

int main(int argc, char* argv[])
{
    // se o comando for dado errado
    if (argc != 2) {
        printf("Uso: %s <nome>\n", argv[0]);
        return 1;
    }

    // Obtém os argumentos
    const char* input_filename = argv[1];

    ///////////////////////////////////////////////////////////
    // WINDOW INITIALIZATION
    ///////////////////////////////////////////////////////////

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
    {
        return -1;
    }

    // Set OpenGL version to 3.3 and core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello Triangle", nullptr, nullptr);

    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;

     // Initialize GLEW
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK && glewErr != GLEW_ERROR_NO_GLX_DISPLAY) {
        // Only fail if the error is NOT the known Wayland/GLX issue
        std::cerr << "FATAL GLEW ERROR: " << glewGetErrorString(glewErr) << std::endl;
        return -1;
    }

    ///////////////////////////////////////////////////////////
    // GEOMETRY DEFINITION - Pyramid with vertices and normals
    ///////////////////////////////////////////////////////////

    // Define pyramid vertices (4 triangular faces + square base)
    ModelData model_data = read_obj(input_filename);  // Read the OBJ file

    std::cout << "Model loaded - Vertex count: " << model_data.vertices.size() 
          << ", Normal count: " << model_data.normals.size() << std::endl;

    if (model_data.normals.empty()) {
        std::cerr << "ERROR: No normals were loaded or calculated!" << std::endl;
    }

    std::vector<glm::vec3> pyramid = model_data.vertices;  // Dynamic size

    // Normal vectors for each vertex (pre-calculated for flat shading)
    std::vector<glm::vec3> pyramidNormals = model_data.normals;

    ///////////////////////////////////////////////////////////
    // BUFFER SETUP - VAO and VBOs
    ///////////////////////////////////////////////////////////

    GLuint VAO; // Vertex Array Object - stores vertex attribute configurations
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO); // Activate this VAO

    // Create 2 Vertex Buffer Objects (VBOs)
    GLuint vbo[2];
    glGenBuffers(2, vbo);

    // First VBO - vertex positions
    glGenBuffers(1, &vbo[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, pyramid.size()*3*sizeof(float), &pyramid[0], GL_STATIC_DRAW);

    // Second VBO - vertex normals
    glGenBuffers(1, &vbo[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, pyramidNormals.size()*3*sizeof(float), &pyramidNormals[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(0); // Enable vertex attribute at index 0 (positions)

    ///////////////////////////////////////////////////////////
    // SHADER SETUP - Vertex and Fragment shaders
    ///////////////////////////////////////////////////////////

    // Vertex shader implements Phong lighting model
    const char* vertexShader =
        "#version 330 core\n"
        "uniform mat4 projMat;" // Projection matrix
        "uniform mat4 mvMat;"   // Model-view matrix

        // Vertex attributes
        "layout (location=0) in vec3 vertPos;"
        "layout (location=1) in vec3 vertNormal;"

        // Output to fragment shader
        "out vec4 varyingColor;"

        // Light properties structure
        "struct PositionalLight"
        "{ vec4 ambient;"
        "vec4 diffuse;"
        "vec4 specular;"
        "vec3 position;"
        "};"

        // Material properties structure
        "struct Material"
        "{ vec4 ambient;"
        "vec4 diffuse;"
        "vec4 specular;"
        "float shininess;"
        "};"

        // Uniform variables
        "uniform vec4 globalAmbient;"  // Global ambient light
        "uniform PositionalLight light;" // Light source
        "uniform Material material;"    // Surface material
        "uniform mat4 norm_matrix;"     // For transforming normals

        "void main()"
        "{"
        // Transform vertex to view space
        "vec4 P =  mvMat*vec4(vertPos,1.0);"

        // Transform and normalize normal vector
        "vec3 N = normalize((norm_matrix * vec4(vertNormal, 1.0)).xyz);"

        // Calculate light direction (L) and view direction (V)
        "vec3 L = normalize(light.position - P.xyz);"
        "vec3 V = normalize(-P.xyz);"

        // Calculate reflection vector (R)
        "vec3 R = reflect(-L, N);"

        // Gouraud lighting calculations:
        // Ambient = global ambient + light's ambient
        "vec3 ambient = ((globalAmbient * material.ambient) + (light.ambient * material.ambient)).xyz;"

        // Diffuse = Lambert's cosine law
        "vec3 diffuse = light.diffuse.xyz * material.diffuse.xyz * max(dot(N,L), 0.0);"

        // Specular = Blinn-Phong highlight
        "vec3 specular= material.specular.xyz * light.specular.xyz * pow(max(dot(R,V), 0.0f), material.shininess);"

        // Combine all components
        "varyingColor = vec4((ambient + diffuse + specular), 1.0);"
        //"varyingColor = vec4(abs(N), 1.0);"   // Mostra as normais como cor

        // Final vertex position
        "gl_Position = (projMat*mvMat)*vec4(vertPos, 1.0);"
        "}";

    // Fragment shader - simply outputs the interpolated color
    const char* fragmentShader =
        "#version 330 core\n"
        "out vec4 color;"
        "in vec4 varyingColor;"
        "void main()"
        "{"
        "color = varyingColor;" // Pass through the color from vertex shader
        "}";

    // Compile and link shaders
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShader, nullptr);
    glCompileShader(vs);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShader, nullptr);
    glCompileShader(fs);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, fs);
    glAttachShader(shaderProgram, vs);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    ///////////////////////////////////////////////////////////
    // PROJECTION MATRIX SETUP
    ///////////////////////////////////////////////////////////

    // Create perspective projection matrix
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    float aspect = static_cast<float>(width)/static_cast<float>(height);

    // Parameters: FOV, aspect ratio, near clip, far clip
    glm::mat4 projMat = glm::perspective(glm::radians(45.f), aspect, 1.0f, 100.0f);
    GLuint projLoc = glGetUniformLocation(shaderProgram, "projMat");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projMat));

    ///////////////////////////////////////////////////////////
    // LIGHTING AND MATERIAL SETUP
    ///////////////////////////////////////////////////////////

    glm::mat4 modelMat, viewMat, mvMat;

    // Light properties
    glm::vec3 currentLightPos, lightPosV; // World and view space positions
    float lightPos[3]; // Will store view space position

    // Light position in world space
    currentLightPos = glm::vec3(0.0f, 10.0f, 0.0f);

    // White light properties
    float globalAmbient[4] = {1.0f, 1.0f, 1.0f, 1.0f }; // Global ambient light

    // Light source properties (white light)
    float lightAmbient[4] = {1.0f, 1.0f, 1.0f, 1.0f };
    float lightDiffuse[4] = {1.0f, 1.0f, 1.0f, 1.0f };
    float lightSpecular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

    // Gold material properties
    float matAmbGold[4] = {0.2473f, 0.1995f, 0.0745f, 1.f}; // Ambient reflectivity
    float matDifGold[4] = {0.7516f, 0.6065f, 0.2265f, 1.f}; // Diffuse reflectivity
    float matSpeGold[4] = {0.6283f, 0.5558f, 0.3661f, 1.f}; // Specular reflectivity
    float matShiGold = 25.f; // Shininess exponent

    // Animation variables
    float modelAngle = 0.f; // Model rotation angle
    float camAngle = 0.f;   // Camera rotation angle

    // Camera position
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);

    ///////////////////////////////////////////////////////////
    // RENDERING LOOP
    ///////////////////////////////////////////////////////////
    while (!glfwWindowShouldClose(window))
    {
        /* Clear the screen */
        glClearColor(0.2, 0.2,0.5,0);
        glClear(GL_COLOR_BUFFER_BIT);

        // Get uniform locations for lighting parameters
        GLuint mvLoc = glGetUniformLocation(shaderProgram, "mvMat");

        // Light uniform locations
        GLuint globalAmbLoc = glGetUniformLocation(shaderProgram, "globalAmbient");
        GLuint ambLoc = glGetUniformLocation(shaderProgram, "light.ambient");
        GLuint diffLoc = glGetUniformLocation(shaderProgram, "light.diffuse");
        GLuint specLoc = glGetUniformLocation(shaderProgram, "light.specular");
        GLuint posLoc = glGetUniformLocation(shaderProgram, "light.position");

        // Material uniform locations
        GLuint mAmbLoc = glGetUniformLocation(shaderProgram, "material.ambient");
        GLuint mDiffLoc = glGetUniformLocation(shaderProgram, "material.diffuse");
        GLuint mSpecLoc = glGetUniformLocation(shaderProgram, "material.specular");
        GLuint mShiLoc = glGetUniformLocation(shaderProgram, "material.shininess");

        /* VIEW MATRIX SETUP */
        glm::vec3 targetPos = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 upDirection = glm::vec3(0.0f, 1.0f, 0.0f);

        // Camera rotation calculations
        auto CTo = glm::translate(glm::mat4(1.f), -cameraPos);
        auto CTb = glm::translate(glm::mat4(1.f), cameraPos);
        auto CR = glm::rotate(glm::mat4(1.f), glm::radians(-camAngle), glm::vec3(0.f, 1.f, 0.f));

        // Calculate rotated camera position
        auto cameraPosR = glm::vec3(glm::vec4(cameraPos, 1.0)*(CTb*CR*CTo));

        // Create view matrix
        glm::mat4 viewMat = glm::lookAt(cameraPosR, targetPos, upDirection);

        /* MODEL MATRIX SETUP */
        // Rotate model around its base
        auto To = glm::translate(glm::mat4(1.f), -pyramid[0]);
        auto S = glm::scale(glm::mat4(1.f), glm::vec3(4.0f)); // Escala 2x
        auto R = glm::rotate(glm::mat4(1.f), glm::radians(modelAngle), glm::vec3(0.f,1.f,0.f));
        auto Tb = glm::translate(glm::mat4(1.f), pyramid[0]);

        modelMat = Tb*R*S*To;

        // Combined model-view matrix
        mvMat = viewMat * modelMat;

        // Send to shader
        glUniformMatrix4fv(mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));

        /* LIGHT POSITION TRANSFORMATION */
        // Convert light position to view space
        lightPosV = glm::vec3(viewMat * glm::vec4(currentLightPos, 1.0));
        lightPos[0] = lightPosV.x;
        lightPos[1] = lightPosV.y;
        lightPos[2] = lightPosV.z;

        /* SET LIGHTING AND MATERIAL UNIFORMS */
        glProgramUniform4fv(shaderProgram, globalAmbLoc, 1, globalAmbient);
        glProgramUniform4fv(shaderProgram, ambLoc, 1, lightAmbient);
        glProgramUniform4fv(shaderProgram, diffLoc, 1, lightDiffuse);
        glProgramUniform4fv(shaderProgram, specLoc, 1, lightSpecular);
        glProgramUniform3fv(shaderProgram, posLoc, 1, lightPos);
        glProgramUniform4fv(shaderProgram, mAmbLoc, 1, matAmbGold);
        glProgramUniform4fv(shaderProgram, mDiffLoc, 1, matDifGold);
        glProgramUniform4fv(shaderProgram, mSpecLoc, 1, matSpeGold);
        glProgramUniform1f(shaderProgram, mShiLoc, matShiGold);

        /* NORMAL MATRIX - For correct normal transformation */
        GLuint nLoc = glGetUniformLocation(shaderProgram, "norm_matrix");
        glm::mat4 invTrMat = glm::transpose(glm::inverse(mvMat));
        glUniformMatrix4fv(nLoc, 1, GL_FALSE, glm::value_ptr(invTrMat));

        /* VERTEX ATTRIBUTE SETUP */
        // Position attribute
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);

        // Normal attribute
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(1);

        // Dentro do rendering loop, antes do glDrawArrays

        // Verifique se os atributos estão ativos
        /*GLint pos_attrib = glGetAttribLocation(shaderProgram, "vertPos");
        GLint norm_attrib = glGetAttribLocation(shaderProgram, "vertNormal");
        std::cout << "Attribute locations - Position: " << pos_attrib 
                << ", Normal: " << norm_attrib << std::endl;

        if (norm_attrib == -1) {
            std::cerr << "ERROR: Normal attribute not found in shader!" << std::endl;
        }*/

        /* DRAW THE PYRAMID */
        glDrawArrays(GL_TRIANGLES, 0, pyramid.size());

        /* SWAP BUFFERS AND POLL EVENTS */
        glfwSwapBuffers(window);
        glfwPollEvents();

        /* UPDATE ANIMATION ANGLES */
        modelAngle = modelAngle<=360 ? ++modelAngle : 0;
        camAngle = camAngle <= 360 ? camAngle+ 0.5 : 0;
    }

    glfwTerminate();
    return 0;
}
