#include "interface.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

// Check if a file with the passed filename exists
bool fileExists(string filename) {
    ifstream inputFile;
    inputFile.open(filename);
    bool isOpen = inputFile.is_open();
    inputFile.close();
    return isOpen;
}

// Get the first unused indexed output filename
string getIndexedFilename() {
    int fileIndex = 1;
    string curFilename = "output1.csv";

    // Run until an unused indexed output filename is found or the file index is too high
    while(fileExists(curFilename) && fileIndex < INT_MAX) {
        fileIndex++;
        curFilename = "output" + to_string(fileIndex) + ".csv";
    }

    // Check if the maximum output filename index has been reached
    if(fileIndex == INT_MAX && fileExists(curFilename)) {
        cerr << "Maximum number of indexed output files used" << endl;
        exit(1);
    }

    return curFilename;
}

// Get program options from the command line
void getOptionsCLI(InputSettings& is, cv::CommandLineParser& parser) {
    is.dictionary = parser.get<int>("d");
    is.showRejected = parser.has("r");
    is.markerLength = parser.get<float>("l");

    if(parser.has("dp")) {
        is.detectorFilename = parser.get<string>("dp");
    }

    if(parser.has("refine")) {
        is.hasRefinement = true;
        is.cornerRefinement = parser.get<int>("refine");
    }

    is.cameraID = parser.get<int>("ci");

    if(parser.has("v")) {
        is.inputFilename = parser.get<string>("v");
    }
    else {
        is.collectionRate = parser.get<int>("cr");
    }

    if(parser.has("o")) {
        is.outputFilename = parser.get<string>("o");
    }
    else {
        is.outputFilename = getIndexedFilename();
    }

    if(parser.has("c")) {
        is.calibFilename = parser.get<string>("c");
    }
}

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

// Create and handle startup GUI widgets
int startupGUIWidgets(InputSettings& is) {
    ImGui::Begin("Options", (bool*) 0, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);

    static bool hasRefinement = false;
    ImGui::Checkbox("Override corner refinement from config file", &hasRefinement);

    static bool readFromFile = false;
    ImGui::Checkbox("Read from file", &readFromFile);

    static bool showRejected = false;
    ImGui::Checkbox("Show rejected candidates", &showRejected);

    const char* cornerRefinements[] = {"CORNER_REFINE_NONE", "CORNER_REFINE_SUBPIX",
                                       "CORNER_REFINE_CONTOUR", "CORNER_REFINE_APRILTAG"};

    // Disable corner refinement input unless overriding config file
    if(hasRefinement == false) {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    static int refinementIndex = 0; // Default is no corner refinement
    ImGui::Combo("Corner refinement", &refinementIndex, cornerRefinements, IM_ARRAYSIZE(cornerRefinements));
    if(hasRefinement == false) {
        ImGui::PopItemFlag();
        ImGui::PopStyleVar();
    }

    const char* dictionaries[] = {"DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000", "DICT_5X5_50", "DICT_5X5_100",
                                  "DICT_5X5_250", "DICT_5X5_1000", "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
                                  "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000", "DICT_ARUCO_ORIGINAL"};
    static int dictionaryIndex = 0; // Default dictionary is DICT_4X4_50
    ImGui::Combo("Dictionary", &dictionaryIndex, dictionaries, IM_ARRAYSIZE(dictionaries));

    // Disable camera ID input if collecting data from file
    if(readFromFile) {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    static int cameraID = 0;
    ImGui::InputInt("Camera ID", &cameraID);
    if(readFromFile) {
        ImGui::PopItemFlag();
        ImGui::PopStyleVar();
    }

    static int collectionRate = 10;
    ImGui::InputInt("Data collections per second", &collectionRate);

    static float markerLength = 0.053f;
    ImGui::InputFloat("Marker length in meters", &markerLength);

    static char calibFilename[128] = "calib.txt";
    ImGui::InputText("Calibration filename", calibFilename, IM_ARRAYSIZE(calibFilename));

    static char detectorFilename[128] = "detector_params.yml";
    ImGui::InputText("Detector parameters filename", detectorFilename, IM_ARRAYSIZE(detectorFilename));

    // Disable input filename text input if not collecting data from file
    if(readFromFile == false) {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    static char inputFilename[128] = "";
    ImGui::InputText("Input filename", inputFilename, IM_ARRAYSIZE(inputFilename));
    if(readFromFile == false) {
        ImGui::PopItemFlag();
        ImGui::PopStyleVar();
    }

    static char outputFilename[128] = "";

    // Check if the output filename in input settings and the text input do not match
    if(strcmp(outputFilename, is.outputFilename.c_str()) != 0) {
        // Copy the default output filename to the GUI once
        strcpy_s(outputFilename, is.outputFilename.c_str());
    }

    ImGui::InputText("Output filename", outputFilename, IM_ARRAYSIZE(outputFilename));

    // Update output filename in input settings
    is.outputFilename = outputFilename;

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(ImColor::HSV(0.4f, 0.6f, 0.6f)));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(ImColor::HSV(0.4f, 0.7f, 0.7f)));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(ImColor::HSV(0.4f, 0.8f, 0.8f)));

    if(ImGui::Button("Start")) {
        is.dictionary = dictionaryIndex;
        is.cornerRefinement = refinementIndex;
        is.hasRefinement = hasRefinement;
        is.showRejected = showRejected;
        is.cameraID = cameraID;
        is.collectionRate = collectionRate;
        is.markerLength = markerLength;
        is.calibFilename = calibFilename;
        is.detectorFilename = detectorFilename;
        is.inputFilename = inputFilename;
        return 1;
    }

    ImGui::SameLine();

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(ImColor::HSV(0.0f, 0.6f, 0.6f)));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(ImColor::HSV(0.0f, 0.7f, 0.7f)));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(ImColor::HSV(0.0f, 0.8f, 0.8f)));

    if(ImGui::Button("Quit")) {
        return -1;
    }

    ImGui::PopStyleColor(6);

    ImGui::End();

    return 0;
}

// Get program options from a GUI
int getOptionsGUI(InputSettings& is) {
    float GUIScalingFactor = 1.5f;

    // 0: Continue running startup GUI, 1: Start data collection, -1: Quit program
    int startCollection = 0;

    is.outputFilename = getIndexedFilename();

    // Set up window
    glfwSetErrorCallback(glfw_error_callback);
    if(!glfwInit())
        return 1;

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(900, 540, "Program Options", NULL, NULL);
    if(window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL loader
    if(gl3wInit() != 0) {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

    // Set up Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void) io;

    // Set up Dear ImGui style
    ImGui::StyleColorsDark();

    // Set up Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    ImGui::GetStyle().ScaleAllSizes(GUIScalingFactor);

    // Scale ImGui font
    ImFontConfig fontConfig;
    constexpr float defaultFontSize = 13.0f;
    fontConfig.SizePixels = defaultFontSize * GUIScalingFactor;
    ImGui::GetIO().Fonts->AddFontDefault(&fontConfig);

    // Main loop
    while(!glfwWindowShouldClose(window) && startCollection == 0) {
        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Make next ImGui window fill OS window
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);

        startCollection = startupGUIWidgets(is);

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    bool stopProgram = false;
    if(startCollection == -1 || glfwWindowShouldClose(window)) {
        stopProgram = true;
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    if(stopProgram) {
        return -1;
    }

    return 0;
}