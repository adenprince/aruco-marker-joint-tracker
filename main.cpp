// ArUco marker detection code obtained from: https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/samples/detect_markers.cpp


#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

// Store program options
struct InputSettings {
    int dictionary = 0;
    int cornerRefinement = 0;
    bool hasRefinement = false;
    bool showRejected = false;
    int cameraID = 0;
    int collectionRate = 0;
    float markerLength = 0.0f;
    string calibFilename;
    string detectorFilename;
    string inputFilename;
    string outputFilename;
};

namespace {
    const char* about = "Basic marker detection";
    const char* keys =
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16,"
        "DICT_APRILTAG_16h5=17, DICT_APRILTAG_25h9=18, DICT_APRILTAG_36h10=19, DICT_APRILTAG_36h11=20}"
        "{v        |       | Input from video file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{c        |       | Camera intrinsic parameters. Needed for camera pose }"
        "{l        | 0.1   | Marker side length (in meters). Needed for correct scale in camera pose }"
        "{dp       |       | File of marker detector parameters }"
        "{r        |       | show rejected candidates too }"
        "{refine   |       | Corner refinement: CORNER_REFINE_NONE=0, CORNER_REFINE_SUBPIX=1,"
        "CORNER_REFINE_CONTOUR=2, CORNER_REFINE_APRILTAG=3}"
        "{o        |       | Joint angle output filename, if none, filename is automatically indexed }"
        "{cr       |       | Number of times per second to collect joint angle data }";
}

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



// Get the angle between two vectors using three passed points
float getJointAngle(vector<Vec3f>& jointPoints) {
    // The second point is the vertex of the angle
    Vec3f v1 = jointPoints.at(0) - jointPoints.at(1);
    Vec3f v2 = jointPoints.at(2) - jointPoints.at(1);
    float angle = acosf(v1.dot(v2) / (norm(v1) * norm(v2))) * 180.0f / (float) CV_PI;

    return angle;
}



/**
 */
static bool readCameraParameters(string filename, Mat& camMatrix, Mat& distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}



/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters>& params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}



// Get program options from the command line
void getOptionsCLI(InputSettings& is, CommandLineParser& parser) {
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
        is.inputFilename = parser.get<String>("v");
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



/**
 */
int main(int argc, char* argv[]) {
    InputSettings is;

    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 2) {
        parser.printMessage();
        return 1;
    }

    getOptionsCLI(is, parser);
    
    bool estimatePose = (is.calibFilename != "");

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if(is.detectorFilename != "") {
        bool readOk = readDetectorParameters(is.detectorFilename, detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 1;
        }
    }

    if(is.hasRefinement) {
        //override cornerRefinementMethod read from config file
        detectorParams->cornerRefinementMethod = is.cornerRefinement;
    }
    cout << "Corner refinement method (0: None, 1: Subpixel, 2:contour, 3: AprilTag 2): " << detectorParams->cornerRefinementMethod << endl;

    // Time between joint angle data collection
    double collectionTime = 0; // Collect data as fast as possible for pre-recorded video

    // Check if there is a collection rate and using real-time video
    if(is.collectionRate != 0 && is.inputFilename == "") {
        collectionTime = 1 / is.collectionRate;
    }

    if(fileExists(is.outputFilename)) {
        cerr << "File " << is.outputFilename << " already exists" << endl;
        return 1;
    }

    if(!parser.check()) {
        parser.printErrors();
        return 1;
    }

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(is.dictionary));

    Mat camMatrix, distCoeffs;
    if(estimatePose) {
        bool readOk = readCameraParameters(is.calibFilename, camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 1;
        }
    }

    ofstream outputFile;
    outputFile.open(is.outputFilename);

    if(outputFile.is_open()) {
        cout << "File " << is.outputFilename << " opened successfully" << endl;
    }
    else {
        cerr << "File " << is.outputFilename << " failed to open" << endl;
        return 1;
    }

    outputFile << "Total Time,Joint Angle" << endl;

    VideoCapture inputVideo;
    int waitTime;
    if(is.inputFilename != "") {
        inputVideo.open(is.inputFilename);
        waitTime = 50;
    }
    else {
        inputVideo.open(is.cameraID);
        waitTime = 10;
    }

    double totalTime = 0;
    int totalIterations = 0;

    double prevCollectionTime = 0;
    double startTime = (double) getTickCount();
    
    while(inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double) getTickCount();

        vector<int> ids;
        vector<vector<Point2f>> corners, rejected;
        vector<Vec3d> rvecs, tvecs;

        // detect markers and estimate pose
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
        if(estimatePose && ids.size() > 0)
            aruco::estimatePoseSingleMarkers(corners, is.markerLength, camMatrix, distCoeffs, rvecs,
                                             tvecs);

        double currentTime = ((double) getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }

        // draw results
        image.copyTo(imageCopy);
        if(ids.size() > 0) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);

            if(estimatePose) {
                vector<Vec3f> jointPoints(3);
                vector<Point2f> jointImagePoints(3);
                float jointAngle = -1.0f;
                int pointsDetected = 0;
                int numIDs = ids.size();

                for(int i = 0; i < numIDs; i++) {
                    float length = is.markerLength * 0.5f;

                    aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i], length);

                    // Get marker 2D image points (code from OpenCV drawFrameAxes function)
                    vector<Point3f> axesPoints;
                    axesPoints.push_back(Point3f(0, 0, 0));
                    axesPoints.push_back(Point3f(length, 0, 0));
                    axesPoints.push_back(Point3f(0, length, 0));
                    axesPoints.push_back(Point3f(0, 0, length));
                    vector<Point2f> imagePoints;
                    projectPoints(axesPoints, rvecs[i], tvecs[i], camMatrix, distCoeffs, imagePoints);

                    int curID = ids.at(i);

                    if(curID < 3) {
                        jointPoints.at(curID) = tvecs[i];
                        jointImagePoints.at(curID) = imagePoints[0];
                        pointsDetected++;
                    }
                }

                if(pointsDetected == 3) {
                    jointAngle = getJointAngle(jointPoints);

                    // Draw joint angle
                    line(imageCopy, jointImagePoints[1], jointImagePoints[0], Scalar(0, 0, 0), 2);
                    line(imageCopy, jointImagePoints[1], jointImagePoints[2], Scalar(0, 0, 0), 2);
                    
                    // Get each line of the joint angle
                    Vec2f v1 = jointImagePoints[0] - jointImagePoints[1];
                    Vec2f v2 = jointImagePoints[2] - jointImagePoints[1];
                    
                    // Get point in the middle of the angle
                    Vec2f bisection = (normalize(v1) + normalize(v2)) * 25.0f;
                    Point2f p;
                    p.x = bisection[0] + jointImagePoints[1].x;
                    p.y = bisection[1] + jointImagePoints[1].y;

                    // Get rounded angle value as a string
                    string displayText = to_string((int) round(jointAngle));
                    
                    // Center angle text
                    int baseline = 0;
                    Size textSize = getTextSize(displayText, 0, 0.5, 2, &baseline);
                    p.x -= textSize.width / 2.0f;
                    p.y -= textSize.height / 2.0f;

                    // Display angle text centered in the angle
                    putText(imageCopy, displayText, p, 0, 0.5, Scalar(255, 255, 255), 2);
                }

                double curTime = ((double) getTickCount() - startTime) / getTickFrequency();

                // Write data to file if enough time has passed or first iteration
                if(curTime - prevCollectionTime >= collectionTime || totalIterations == 1) {
                    if(pointsDetected == 3) {
                        outputFile << curTime << "," << jointAngle << endl;
                    }
                    else {
                        outputFile << curTime << "," << endl;
                    }

                    prevCollectionTime = curTime;
                }
            }
        }

        if(is.showRejected && rejected.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));

        imshow("out", imageCopy);
        char key = (char) waitKey(waitTime);
        if(key == 27) break;
    }

    return 0;
}