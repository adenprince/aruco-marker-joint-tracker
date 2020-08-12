/* Aden Prince
 * HiMER Lab at U. of Illinois, Chicago
 * ArUco Marker Joint Tracker
 * 
 * interface.cpp
 * Contains functions for getting program options and displaying program usage.
 * 
 * ImGui sample code obtained from: https://github.com/ocornut/imgui/blob/master/examples/example_win32_directx11/main.cpp
 * ImGui font scaling code obtained from: https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/tools/k4aviewer/k4aviewer.cpp
 */

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
void getOptionsCLI(InputSettings& is, CommandLineParser& parser) {
    // Getting an option that does not exist throws an error
    is.dictionary = parser.get<int>("d");
    is.showRejected = parser.has("r");
    is.markerLength = parser.get<float>("l");

    // Check if there is a --dp flag before getting its value (flag is optional)
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
        // Get collection rate if not collecting data from a video file
        is.collectionRate = parser.get<int>("cr");
    }

    if(parser.has("o")) {
        is.outputFilename = parser.get<string>("o");
    }
    else {
        // Set output filename to default if not given in the command line
        is.outputFilename = getIndexedFilename();
    }

    if(parser.has("c")) {
        is.calibFilename = parser.get<string>("c");
    }

    is.numJoints = parser.get<int>("j");
}

// Display an error message when a GLFW error occurs
static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

// Create and handle startup GUI widgets
int startupGUIWidgets(InputSettings& is, string& errorText) {
    // 0: Continue running startup GUI, 1: Start data collection, -1: Quit program
    int startCollection = 0;

    static bool hasRefinement = false;
    ImGui::Checkbox("Override corner refinement from config file", &hasRefinement);

    static bool readFromFile = false;
    ImGui::Checkbox("Read from file", &readFromFile);

    static bool showRejected = false;
    ImGui::Checkbox("Show rejected candidates", &showRejected);

    const char* cornerRefinements[] = {"CORNER_REFINE_NONE", "CORNER_REFINE_SUBPIX",
                                       "CORNER_REFINE_CONTOUR", "CORNER_REFINE_APRILTAG"};

    // Disable corner refinement input unless overriding config file
    if(!hasRefinement) {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    static int refinementIndex = 0; // Default is no corner refinement
    ImGui::Combo("Corner refinement", &refinementIndex, cornerRefinements, IM_ARRAYSIZE(cornerRefinements));
    if(!hasRefinement) {
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

    static int numJoints = 1;
    ImGui::InputInt("Number of joints", &numJoints);

    static float markerLength = 0.053f;
    ImGui::InputFloat("Marker length in meters", &markerLength);

    static char calibFilename[128] = "calib.txt";
    ImGui::InputText("Calibration filename", calibFilename, IM_ARRAYSIZE(calibFilename));

    static char detectorFilename[128] = "detector_params.yml";
    ImGui::InputText("Detector parameters filename", detectorFilename, IM_ARRAYSIZE(detectorFilename));

    // Disable input filename text input if not collecting data from file
    if(!readFromFile) {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    static char inputFilename[128] = "";
    ImGui::InputText("Input filename", inputFilename, IM_ARRAYSIZE(inputFilename));
    if(!readFromFile) {
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

    // Make Start button green
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(ImColor::HSV(0.4f, 0.6f, 0.6f)));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(ImColor::HSV(0.4f, 0.7f, 0.7f)));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(ImColor::HSV(0.4f, 0.8f, 0.8f)));

    if(ImGui::Button("Start")) {
        // Reset error text
        errorText = "";

        // Update values in inputSettings
        is.dictionary = dictionaryIndex;
        is.cornerRefinement = refinementIndex;
        is.hasRefinement = hasRefinement;
        is.showRejected = showRejected;
        is.cameraID = cameraID;
        is.collectionRate = collectionRate;
        is.numJoints = numJoints;
        is.markerLength = markerLength;
        is.calibFilename = calibFilename;
        is.detectorFilename = detectorFilename;
        is.inputFilename = inputFilename;

        // 1 is returned and data collection starts if there are no errors
        startCollection = 1;

        // Check for errors
        if(!readFromFile && cameraID < 0) {
            errorText += "ERROR: Camera ID cannot be negative\n";
            startCollection = 0;
        }

        if(collectionRate < 0) {
            errorText += "ERROR: Data collection rate cannot be negative\n";
            startCollection = 0;
        }

        if(numJoints < 0) {
            errorText += "ERROR: Number of joints cannot be negative\n";
            startCollection = 0;
        }

        if(markerLength <= 0) {
            errorText += "ERROR: Marker length must be positive\n";
            startCollection = 0;
        }

        // Check if there are no non-space characters in the calibration filename
        if(is.calibFilename.find_first_not_of(' ') != string::npos &&
           !fileExists(is.calibFilename)) {
            errorText += "ERROR: Calibration file \"" + is.calibFilename + "\" not found\n";
            startCollection = 0;
        }

        if(is.detectorFilename.find_first_not_of(' ') != string::npos &&
           !fileExists(is.detectorFilename)) {
            errorText += "ERROR: Detector parameters file \"" + is.detectorFilename + "\" not found\n";
            startCollection = 0;
        }

        if(readFromFile && !fileExists(is.inputFilename)) {
            errorText += "ERROR: Input file \"" + is.inputFilename + "\" not found\n";
            startCollection = 0;
        }

        if(is.outputFilename.find_first_not_of(' ') == string::npos) {
            errorText += "ERROR: Output filename is empty\n";
            startCollection = 0;
        }

        if(fileExists(is.outputFilename)) {
            errorText += "ERROR: Output file \"" + is.outputFilename + "\" already exists\n";
            startCollection = 0;
        }
    }

    ImGui::SameLine();

    // Make Quit button red
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(ImColor::HSV(0.0f, 0.6f, 0.6f)));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(ImColor::HSV(0.0f, 0.7f, 0.7f)));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(ImColor::HSV(0.0f, 0.8f, 0.8f)));

    if(ImGui::Button("Quit")) {
        startCollection = -1;
    }

    // Display error messages in red
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.0f, 1.0f));
    ImGui::TextWrapped(errorText.c_str());

    // Remove style settings
    ImGui::PopStyleColor(7);

    return startCollection;
}

// Get program options from a GUI
int getOptionsGUI(InputSettings& is) {
    float GUIScalingFactor = 1.5f;

    // 0: Continue running startup GUI, 1: Start data collection, -1: Quit program
    int startCollection = 0;

    string errorText = "";

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
    GLFWwindow* window = glfwCreateWindow(910, 650, "Program Options", NULL, NULL);
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

        // Create options window with startup widgets
        ImGui::Begin("Options", (bool*) 0, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
        startCollection = startupGUIWidgets(is, errorText);
        ImGui::End();

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
        stopProgram = true; // Stop program if options window was closed
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    if(stopProgram) {
        return -1; // Indicates that the program should stop
    }

    return 0;
}