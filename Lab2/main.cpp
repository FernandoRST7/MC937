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

/*bibliotecas utilizadas: flm, glew, glfw e as padrao do c++*/

std::vector<glm::vec3> generateVibrantRandomColors(const std::vector<glm::vec3>& vertices) {
    std::vector<glm::vec3> colors;
    colors.reserve(vertices.size());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 2); // For selecting which component to emphasize
    std::uniform_real_distribution<float> colorDis(0.5f, 1.0f); // Keep colors bright

    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec3 color(0.2f, 0.2f, 0.2f); // Base color (dark)
        int primary = dis(gen); // Choose which component to emphasize (0=R, 1=G, 2=B)
        color[primary] = colorDis(gen);
        colors.push_back(color);
    }

    return colors;
}

std::vector<glm::vec3> read_obj(const char* input_filename){
    /*Opens an obj file, saves the vectors, faces and normals, creates the verticces vector and returns it*/
    // Open the file
    std::ifstream file(input_filename);

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Failed to open the file!" << std::endl;
    }

    std::vector<glm::vec3> vertices;  // Dynamic size
    
    std::vector<glm::vec3> verticeVetor;    // Will store n1, n2, n3 coordinates sequentially
    std::vector<glm::vec3> faceVetor;       // Will store n1, n2, n3 vertices of the triangle sequentially
    std::vector<glm::vec3> normaVetor;      // Will store n1, n2, n3 coordinates sequentially

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {  // Vertex line: "v x=n1 y=n2 z=n3"
            //save the float points
            float n1, n2, n3;
            iss >> n1 >> n2 >> n3;

            verticeVetor.push_back(glm::vec3(n1, n2, n3));

        } else if (prefix == "f") {
            //save the order of the points
            std::string v1, v2, v3;
            iss >> v1 >> v2 >> v3;

            // Extract just the vertex index (handling formats like "1", "1/2", "1/2/3")
            unsigned int n1 = std::stoi(v1.substr(0, v1.find_first_of('/')));
            unsigned int n2 = std::stoi(v2.substr(0, v2.find_first_of('/')));
            unsigned int n3 = std::stoi(v3.substr(0, v3.find_first_of('/')));

            faceVetor.push_back(glm::vec3(n1, n2, n3));  

        } else if (prefix == "vn") {  // Vertex line: "v x y z"
            //save the float points
            float n1, n2, n3;
            iss >> n1 >> n2 >> n3;

            verticeVetor.push_back(glm::vec3(n1, n2, n3));
        } else { continue;}
        
    }

    for (const auto& face : faceVetor) {                    // for each face (has 3 vertix), add 3 float points
        for (int i=0; i<3; i++){                            // does it for first(0), sec(1) and third(2) vertex of the face
            int i_ = face[i] - 1;                           // index of the vertice of the triangle
                vertices.push_back(verticeVetor[i_]);       // adds vertice of the triangle (x=n1)
        }
    }

    return vertices;

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
    window = glfwCreateWindow(640, 480, "Bunny rendered", nullptr, nullptr);

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
    std::vector<glm::vec3> coords = read_obj(input_filename);  // Dynamic size
    std::vector<glm::vec3> colors = generateVibrantRandomColors(coords);

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
    glm::mat4 R = glm::rotate(glm::mat4(1.f),  glm::radians(-30.f), glm::vec3(0.f,1.f,0.f));
    glm::mat4 Tb = glm::translate(glm::mat4(1.f), coords[0]);  // Move back

    auto model = Tb * S * R * To;  // Final transformation matrix

    GLuint transformLoc = glGetUniformLocation(shaderProgram, "transform");

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {

        /* Render here */
        glClearColor(0.0f, 0.0f, 0.5f, 0.4f);
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
