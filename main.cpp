/* Aden Prince
 * HiMER Lab at U. of Illinois, Chicago
 * ArUco Marker Joint Tracker
 * 
 * main.cpp
 * Gets inputs settings and runs data collection.
 * 
 * ArUco marker detection code obtained from: https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/samples/detect_markers.cpp
 */

#include "interface.h"
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

namespace {
    const char* about = "Basic marker detection";
    const char* keys =
        "{h        |       | Display help information }"
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
        "{cr       |       | Number of times per second to collect joint angle data }"
        "{j        | 1     | Number of joints to collect angle data for }";
}

// Function from OpenCV library
// Converts a given Rotation Matrix to Euler angles
// Convention used is X-Y-Z Tait-Bryan angles
// Reference code implementation:
// https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
Vec3f rot2euler(const cv::Mat& rotationMatrix) {
    Vec3f euler;

    double m00 = rotationMatrix.at<double>(0, 0);
    double m02 = rotationMatrix.at<double>(0, 2);
    double m10 = rotationMatrix.at<double>(1, 0);
    double m11 = rotationMatrix.at<double>(1, 1);
    double m12 = rotationMatrix.at<double>(1, 2);
    double m20 = rotationMatrix.at<double>(2, 0);
    double m22 = rotationMatrix.at<double>(2, 2);

    double bank, attitude, heading;

    // Assuming the angles are in radians.
    if(m10 > 0.998) { // singularity at north pole
        bank = 0;
        attitude = CV_PI / 2;
        heading = atan2(m02, m22);
    }
    else if(m10 < -0.998) { // singularity at south pole
        bank = 0;
        attitude = -CV_PI / 2;
        heading = atan2(m02, m22);
    }
    else {
        bank = atan2(-m12, m11);
        attitude = asin(m10);
        heading = atan2(-m20, m00);
    }

    euler[0] = bank * 180.0f / (float) CV_PI;
    euler[1] = heading * 180.0f / (float) CV_PI;
    euler[2] = attitude * 180.0f / (float) CV_PI;

    return euler;
}

// Get the angle between two vectors using three passed points
float getJointAngle(vector<Vec3f>& jointPoints, size_t startIndex) {
    // The second point is the vertex of the angle
    Vec3f v1 = jointPoints.at(startIndex) - jointPoints.at(startIndex + 1);
    Vec3f v2 = jointPoints.at(startIndex + 2) - jointPoints.at(startIndex + 1);
    float angle = acosf(v1.dot(v2) / (norm(v1) * norm(v2))) * 180.0f / (float) CV_PI;

    return angle;
}

// Read camera parameters from a given file and store them in passed variables
static bool readCameraParameters(string filename, Mat& camMatrix, Mat& distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

// Read detector parameters from a given file and store them in passed variables
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

int main(int argc, char* argv[]) {
    InputSettings is;

    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    // Run startup GUI if no command-line options are given
    if(argc < 2) {
        int resultGUI = getOptionsGUI(is);

        if(resultGUI == 1) {
            // Return error value
            return 1;
        }
        else if(resultGUI == -1) {
            // Exit program
            return 0;
        }
    }
    else {
        // Display program info if there is a -h flag
        if(parser.has("h")) {
            parser.printMessage();
            return 0;
        }

        getOptionsCLI(is, parser);
    }
    
    // Estimate marker pose if a camera calibration file is given
    bool estimatePose = (is.calibFilename != "");

    // Read detector parameters file
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if(is.detectorFilename != "") {
        bool readOk = readDetectorParameters(is.detectorFilename, detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 1;
        }
    }

    if(is.hasRefinement) {
        // Override cornerRefinementMethod read from config file
        detectorParams->cornerRefinementMethod = is.cornerRefinement;
    }
    cout << "Corner refinement method (0: None, 1: Subpixel, 2:contour, 3: AprilTag 2): " << detectorParams->cornerRefinementMethod << endl;

    // Time between joint angle data collection
    double collectionTime = 0; // Collect data as fast as possible for pre-recorded video

    // Check if there is a collection rate and using real-time video
    if(is.collectionRate != 0 && is.inputFilename == "") {
        collectionTime = 1.0f / is.collectionRate;
    }

    if(fileExists(is.outputFilename)) {
        cerr << "File " << is.outputFilename << " already exists" << endl;
        return 1;
    }

    // Check for command-line option errors
    if(!parser.check()) {
        parser.printErrors();
        return 1;
    }

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(is.dictionary));

    // Read camera calibration file
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
        cout << "File \"" << is.outputFilename << "\" opened successfully" << endl;
    }
    else {
        cerr << "File \"" << is.outputFilename << "\" failed to open" << endl;
        return 1;
    }

    // Print column titles to data output file
    outputFile << "Total Time";
    for(int i = 1; i <= is.numJoints; ++i) {
        outputFile << ",Joint " << i << " Angle";
    }
    for(int i = 0; i < is.numJoints + 2; ++i) {
        outputFile << ",Marker " << i << " Rotation";
    }
    outputFile << endl;

    // Get video input from either a file or a camera
    VideoCapture inputVideo;
    if(is.inputFilename != "") {
        inputVideo.open(is.inputFilename);
    }
    else {
        inputVideo.open(is.cameraID);
    }

    double totalDetectionTime = 0;
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

        // Detect markers and estimate pose
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
        if(estimatePose && ids.size() > 0)
            aruco::estimatePoseSingleMarkers(corners, is.markerLength, camMatrix, distCoeffs, rvecs,
                                             tvecs);

        double currentTime = ((double) getTickCount() - tick) / getTickFrequency();
        totalDetectionTime += currentTime;
        ++totalIterations;

        // Output detection time info every 30 loop iterations
        if(totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalDetectionTime / double(totalIterations)
                 << " ms)" << endl;
        }

        vector<float> jointAngles((size_t) is.numJoints);
        vector<bool> anglesDetected((size_t) is.numJoints);
        vector<bool> pointsDetected((size_t) is.numJoints + 2);
        vector<Vec3f> markerAngles((size_t) is.numJoints + 2);

        // draw results
        image.copyTo(imageCopy);
        if(ids.size() > 0) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);

            if(estimatePose) {
                vector<Vec3f> jointPoints((size_t) is.numJoints + 2);
                vector<Point2f> jointImagePoints((size_t) is.numJoints + 2);
                int numIDs = ids.size();

                for(int i = 0; i < numIDs; ++i) {
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

                    int curID = ids[i];

                    // Collect marker data if its ID is in the correct range
                    if(curID < is.numJoints + 2) {
                        jointPoints[curID] = tvecs[i];
                        jointImagePoints[curID] = imagePoints[0];
                        pointsDetected[curID] = true;

                        Mat rotationMatrix;
                        Rodrigues(rvecs[i], rotationMatrix);
                        markerAngles[curID] = rot2euler(rotationMatrix);
                    }
                }

                // Draw each joint angle
                for(size_t i = 0; i < is.numJoints; ++i) {
                    // Check that the points needed for the current angle are detected
                    anglesDetected[i] = (pointsDetected[i] && pointsDetected[i + 1] && pointsDetected[i + 2]);

                    if(anglesDetected[i]) {
                        jointAngles[i] = getJointAngle(jointPoints, i);

                        // Draw joint angle lines
                        if(i == 0 || !anglesDetected[i - 1]) {
                            // Draw first line if it has not been drawn for a previous angle
                            line(imageCopy, jointImagePoints[i + 1], jointImagePoints[i], Scalar(0, 0, 0), 2);
                        }
                        line(imageCopy, jointImagePoints[i + 1], jointImagePoints[i + 2], Scalar(0, 0, 0), 2);

                        // Get each line of the joint angle
                        Vec2f v1 = jointImagePoints[i] - jointImagePoints[i + 1];
                        Vec2f v2 = jointImagePoints[i + 2] - jointImagePoints[i + 1];

                        // Get point in the middle of the angle
                        Vec2f bisection = (normalize(v1) + normalize(v2)) * 25.0f;
                        Point2f p;
                        p.x = bisection[0] + jointImagePoints[i + 1].x;
                        p.y = bisection[1] + jointImagePoints[i + 1].y;

                        // Get rounded angle value as a string
                        string displayText = to_string((int) round(jointAngles[i]));

                        // Center angle text
                        int baseline = 0;
                        Size textSize = getTextSize(displayText, 0, 0.5, 2, &baseline);
                        p.x -= textSize.width / 2.0f;
                        p.y -= textSize.height / 2.0f;

                        // Display angle text centered in the angle
                        putText(imageCopy, displayText, p, 0, 0.5, Scalar(255, 255, 255), 2);
                    }
                }
            }
        }

        currentTime = ((double) getTickCount() - startTime) / getTickFrequency();

        // Write data to file if enough time has passed or first iteration
        if(currentTime - prevCollectionTime >= collectionTime || totalIterations == 1) {
            // Write program run time
            outputFile << currentTime;

            // Write joint angle data
            for(int i = 0; i < is.numJoints; ++i) {
                outputFile << ",";
                if(anglesDetected[i]) {
                    outputFile << jointAngles[i];
                }
            }

            // Write marker rotation data
            for(int i = 0; i < is.numJoints + 2; ++i) {
                outputFile << ",";
                if(pointsDetected[i]) {
                    outputFile << "\"" << markerAngles[i][0] << "," << markerAngles[i][1]
                               << "," << markerAngles[i][2] << "\"";
                }
            }

            outputFile << endl;

            prevCollectionTime = currentTime;
        }

        // Draw rejected marker candidates if needed
        if(is.showRejected && rejected.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));

        // Show camera view window with drawn information
        imshow("Camera View", imageCopy);

        // Get keyboard input and stop program when the Esc key is pressed
        char key = (char) waitKey(1);
        if(key == 27) break;
    }

    return 0;
}