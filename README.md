# ArUco Marker Joint Tracker

This program uses ArUco markers to track joints, such as on a robotic arm, and write their angle data to a CSV (comma-separated values) file. Video data can come from a camera or a pre-recorded video file.

## Usage

For a selected ArUco dictionary, the marker with ID 0 should be at the base of the joint, and the marker IDs should increase by 1 for each point to be tracked. A startup GUI is displayed if no command-line options are given. Use the -h flag to display how to set options through the command line. Program options include:

 - Show rejected marker candidates
 - Corner refinement
 - ArUco marker dictionary
 - Camera ID
 - Angle data collections per second
 - Number of joints
 - ArUco marker length in meters
 - Camera calibration filename
 - Marker detector parameters filename
 - Input video filename
 - Output angle data filename