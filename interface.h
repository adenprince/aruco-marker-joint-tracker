/* Aden Prince
 * HiMER Lab at U. of Illinois, Chicago
 * ArUco Marker Joint Tracker
 * 
 * interface.h
 * Contains code used in multiple source files, such as libraries,
 * function declarations, and an InputSettings structure definition.
 */

#include <opencv2/highgui.hpp>
#include <string>

// Store program options
struct InputSettings {
    int dictionary = 0;
    int cornerRefinement = 0;
    bool hasRefinement = false;
    bool showRejected = false;
    int cameraID = 0;
    int collectionRate = 0;
    int numJoints = 0;
    float markerLength = 0.0f;
    std::string calibFilename;
    std::string detectorFilename;
    std::string inputFilename;
    std::string outputFilename;
};

// Check if a file with the passed filename exists
bool fileExists(std::string filename);
// Get program options from the command line
void getOptionsCLI(InputSettings& is, cv::CommandLineParser& parser);
// Get program options from a GUI
int getOptionsGUI(InputSettings& is);