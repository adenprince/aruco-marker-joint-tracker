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

bool fileExists(std::string filename);
void getOptionsCLI(InputSettings& is, cv::CommandLineParser& parser);
int getOptionsGUI(InputSettings& is);