#include <iostream>
#include <iomanip>
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

/*bibliotecas utilizadas: flm, glew, glfw e as padrao do c++*/

void updateProgressBar(float progress, int barWidth = 50) {
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i <= pos) std::cout << "#";
        else std::cout << "-";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

struct ModelData {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> triangles; // Adiciona vetor de triângulos
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
    std::vector<glm::vec3> triangles;

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
            
            if (has_predefined_normals) {
                faces_normal_indices.emplace_back(n_idx1, n_idx2, n_idx3);
            }
        } else {
            continue; // Ignora outras linhas
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
            model_data.triangles.push_back(glm::vec3(idx0, idx1, idx2));

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

std::vector<glm::vec3> generate_spherical_rays(
    float theta_begin, float theta_end,
    float phi_begin, float phi_end,
    int num_rays) {
    
    // Gera raios esféricos uniformemente distribuídos
    std::vector<glm::vec3> ray_directions;
    ray_directions.reserve(num_rays * num_rays); // Pré-aloca espaço

    const float theta_step = (theta_end - theta_begin) / (num_rays - 1);
    const float phi_step = (phi_end - phi_begin) / (num_rays - 1);

    for (int i = 0; i < num_rays; ++i) {
        float phi = phi_begin + i * phi_step;
        
        for (int j = 0; j < num_rays; ++j) {
            float theta = theta_begin + j * theta_step;
            
            // Conversão de coordenadas esféricas para cartesianas
            float x = std::sin(phi) * std::cos(theta);
            float y = std::sin(phi) * std::sin(theta);
            float z = std::cos(phi);
            
            glm::vec3 direction(x, y, z);
            direction = glm::normalize(direction); // Normalização
            ray_directions.push_back(direction);
        }
    }

    return ray_directions;
}


std::optional<glm::vec3> ray_triangle_intersection(
    const glm::vec3& ray_orig,
    const glm::vec3& ray_dir,
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    float epsilon = 1e-6f)
{
    // Passo 1: Vetores das arestas do triângulo
    const glm::vec3 e1 = v1 - v0;
    const glm::vec3 e2 = v2 - v0;

    // Passo 2: Calcula vetor perpendicular e determinante
    const glm::vec3 p = glm::cross(ray_dir, e2);
    const float det = glm::dot(e1, p);

    // Verifica se o raio é paralelo ao triângulo
    if (std::abs(det) < epsilon) {
        return std::nullopt;
    }

    const float inv_det = 1.0f / det;

    // Passo 3: Calcula parâmetro u e testa limites
    const glm::vec3 t = ray_orig - v0;
    const float u = glm::dot(t, p) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        return std::nullopt;
    }

    // Passo 4: Calcula vetor Q e parâmetro v
    const glm::vec3 q = glm::cross(t, e1);
    const float v = glm::dot(ray_dir, q) * inv_det;
    if (v < 0.0f || (u + v) > 1.0f) {
        return std::nullopt;
    }

    // Passo 5: Calcula t (distância ao ponto de interseção)
    const float t_dist = glm::dot(e2, q) * inv_det;
    if (t_dist >= epsilon) {
        // Retorna ponto de interseção
        return ray_orig + ray_dir * t_dist;
    }

    return std::nullopt;
}

std::vector<glm::vec3> calculate_vertex_colors(
    ModelData model_data,
    const glm::vec3& light_pos,
    const glm::vec3& camera_pos,
    int ray_samples = 16)
{
    const std::vector<glm::vec3>& vertices = model_data.vertices;
    const std::vector<glm::vec3>& normals = model_data.normals;
    const std::vector<glm::vec3>& triangles = model_data.triangles;
    std::vector<glm::vec3> vertex_colors;
    vertex_colors.reserve(vertices.size());

    // Propriedades da luz
    glm::vec3 light_ambient(1.0f, 1.0f, 1.0f);
    glm::vec3 light_diffuse(1.0f, 1.0f, 1.0f);
    glm::vec3 light_specular(1.0f, 1.0f, 1.0f);

    // Propriedades do material (emerald)
    glm::vec3 mat_ambient(0.0215f, 0.1745f, 0.0215f);
    glm::vec3 mat_diffuse(0.07568f, 0.61424f, 0.07568f);
    glm::vec3 mat_specular(0.633f, 0.727811f, 0.633f);
    float mat_shininess = 76.8f;

    std::cout << "Calculando iluminação por vértice:\n";
    for (size_t i = 0; i < vertices.size(); i++) {
        updateProgressBar(float(i) / vertices.size(), 100);
        const auto& vertex = vertices[i];
        const auto& normal = glm::normalize(normals[i]);
        
        // Teste de oclusão
        bool is_occluded = false;
        glm::vec3 ray_dir = glm::normalize(light_pos - vertex); // Direção do vértice para a luz
        
        // itera de 3 em 3, pois cada face eh composta por 3 vertices
        for (size_t j = 0; j < vertices.size(); j += 3) {
            auto hit = ray_triangle_intersection(
                vertex, ray_dir, // Origem no vértice
                vertices[j], vertices[j+1], vertices[j+2]
            );
            if (hit && glm::distance(vertex, *hit) < glm::distance(vertex, light_pos)) {
                is_occluded = true;
                break;
            }
        }

        // if ocluded -> black
        glm::vec3 color(0.0f, 0.0f, 0.0f);
        // else -> ADS
        if (!is_occluded) {
            // Cálculo ADS
            color = light_ambient * mat_ambient; // Componente ambiente sempre aplicada
            glm::vec3 light_dir = glm::normalize(light_pos - vertex);
            // Componente difusa
            float diff = std::max(glm::dot(normal, light_dir), 0.0f);
            color += light_diffuse * mat_diffuse * diff;

            // Componente especular
            glm::vec3 view_dir = glm::normalize(camera_pos - vertex);
            glm::vec3 reflect_dir = glm::reflect(-light_dir, normal);
            float spec = pow(std::max(glm::dot(view_dir, reflect_dir), 0.0f), mat_shininess);
            color += light_specular * mat_specular * spec;
        }
        
        vertex_colors.push_back(color);
    }
    std::cout << "\n";
    
    return vertex_colors;
}

std::vector<glm::vec3> calculate_triangle_colors(
    const std::vector<glm::vec3>& vertex_colors)
{
    std::vector<glm::vec3> triangle_colors;
    std::cout << "Calculando iluminação por face:\n";
    for (size_t i = 0; i < vertex_colors.size(); i += 3) {
        updateProgressBar(float(i) / vertex_colors.size());
        glm::vec3 avg_color = (
            vertex_colors[i] + 
            vertex_colors[i+1] + 
            vertex_colors[i+2]
        ) / 3.0f;
        triangle_colors.push_back(avg_color);
    }
    std::cout << "\n";
    return triangle_colors;
}

void write_mtl_file(
    const std::string& filename,
    const std::vector<glm::vec3>& triangle_colors)
{
    std::ofstream file(filename);
    for (size_t i = 0; i < triangle_colors.size(); i++) {
        const auto& color = triangle_colors[i];
        file << "newmtl material_" << i << "\n"
             << "Kd " << color.r << " " << color.g << " " << color.b << "\n"
             << "\n";
    }
    file.close();
}

void write_obj_file(
    const std::string& filename,
    const std::vector<glm::vec3>& vertices,
    const std::vector<glm::vec3>& triangle_colors)
{
    std::ofstream file(filename);
    file << "mtllib output.mtl\n";
    
    // Escreve vértices
    for (const auto& v : vertices) {
        file << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }
    
    // Escreve faces com materiais
    for (size_t i = 0; i < triangle_colors.size(); i++) {
        size_t base_idx = i * 3;
        file << "usemtl material_" << i << "\n"
             << "f " << base_idx+1 << " " << base_idx+2 << " " << base_idx+3 << "\n";
    }
    file.close();
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

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
    {
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Model rendered", nullptr, nullptr);

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

    
    // OpenGL begins here
    
    // Vertex data for a triangle

    //GLfloat vertices[faceVetor.size()] = {};
    ModelData model_data = read_obj(input_filename);
    std::vector<glm::vec3> coords = model_data.vertices;
    std::vector<glm::vec3> normals = model_data.normals;

    // 1. Calcular cores via ray casting
    glm::vec3 light_pos(0.0f, 10.0f, 0.0f);
    glm::vec3 camera_pos = glm::vec3(0.0f, 0.0f, 3.0f);

    auto vertex_colors = calculate_vertex_colors(model_data, light_pos, camera_pos, 50);
    
    // 2. Calcular cores por triângulo
    auto triangle_colors = calculate_triangle_colors(vertex_colors);
    
    // 3. Gerar arquivos de saída
    // No pdf pede para atualizar, mas optei por gerar um novo para ainda ter o original
    write_mtl_file("output.mtl", triangle_colors);
    write_obj_file("output.obj", coords, triangle_colors);
    
    // 4. [Opcional] Usar cores calculadas para renderização
    std::vector<glm::vec3> colors = vertex_colors; // Para visualização Gouraud

    /*
    previamente, usava-se essa função para definir as cores dos vértices,
    agora deve ser utilizado ray casting com gouraud shading
    std::vector<glm::vec3> colors = generateVibrantRandomColors(coords);
    */
    /*
    GLfloat vertices[] = {
        0.0f, 0.5f, 0.0f, // Top vertex
        -0.5f, -0.5f, 0.0f,  // Bottom-left vertex
        0.5f, -0.5f, 0.0f   // Bottom-right vertex
    };
    */

    // Create VAO and VBO for the triangle

    GLuint VAO; //Stores the configuration of how vertex data is read from the VBO.
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO); //make it active


    GLuint VBO[2]; //Stores vertex data (positions, colors, texture coordinates, etc.) in GPU memory.
    glGenBuffers(2, VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]); //make it active

    //Defines how vertex data is structured
    glVertexAttribPointer(0,// Attribute location in the shader
                          3,//Number of components per vertex
                          GL_FLOAT,//Data type of each component
                          GL_FALSE,//Normalize integer values to [0,1]
                          0,//Byte offset between consecutive attributes
                          nullptr);//Offset from the start of VBO where data begins;


    // colors
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]); //make it active
    //Defines how vertex data is structured
    glVertexAttribPointer(1,// Attribute location in the shader
                          3,//Number of components per vertex
                          GL_FLOAT,//Data type of each component
                          GL_FALSE,//Normalize integer values to [0,1]
                          0,//Byte offset between consecutive attributes
                          nullptr);//Offset from the start of VBO where data begins;



    const char * vertex_shader =
        "#version 330 core\n"
        "layout (location=0) in vec3 coord;"
        "layout (location=1) in vec3 color;"
        "out vec3 outColor;"
        "uniform mat4 transform;"
        "void main(){"
        "gl_Position = transform*vec4(coord, 1.0);"
        "outColor = color;"
        "}";

    const char * fragmet_shader =
        "#version 330 core\n"
        "out vec4 frag_color;"
        "in vec3 outColor;"
        "void main(){"
        "frag_color = vec4(outColor, 0.5);"
        "}";



    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertex_shader, nullptr);
    glCompileShader(vs);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmet_shader, nullptr);
    glCompileShader(fs);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, fs);
    glAttachShader(shaderProgram, vs);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);


    glm::mat4 To = glm::translate(glm::mat4(1.f), -coords[0]);  // Move to origin
    glm::mat4 S = glm::scale(glm::mat4(1.f), glm::vec3(7.0f));  // Apply scaling
    glm::mat4 R = glm::rotate(glm::mat4(1.f),  glm::radians(-60.f), glm::vec3(0.f,1.f,0.f));
    glm::mat4 Tb = glm::translate(glm::mat4(1.f), coords[0]);  // Move back

    auto model = Tb * S * R * To;  // Final transformation matrix

    GLuint transformLoc = glGetUniformLocation(shaderProgram, "transform");

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {

        /* Render here */
        glClearColor(0.0f, 0.2f, 0.3f, 0.4f);
        glClear(GL_COLOR_BUFFER_BIT);


        // Upload coords
        glBindBuffer(GL_ARRAY_BUFFER, VBO[0]); //make it active
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*coords.size(), &coords[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); //Enables the attribute (position)
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(model) );

        glDrawArrays(GL_TRIANGLES, 0, coords.size());

        //upload colors
        glBindBuffer(GL_ARRAY_BUFFER, VBO[1]); //make it active
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*colors.size(), &colors[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(1); //Enables the attribute (position)


        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    std::cout << "EOF" << std::endl;
    return 0;
}
