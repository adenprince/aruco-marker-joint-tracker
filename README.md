# ArUco Marker Joint Tracker

This program uses three ArUco markers to track a joint, such as a robotic arm, and write its angle data to a CSV (comma-separated values) file. Video data can come from a camera or a pre-recorded video file.

## Usage

For a selected ArUco dictionary, the marker with ID 0 should be at the base of the joint, ID 1 should be at the middle of the joint, and ID 2 should be at the end of the joint. A startup GUI is displayed if no command-line options are given. Use the -h flag to display how to set options through the command line. Program options include:

 - Show rejected marker candidates
 - Corner refinement
 - ArUco marker dictionary
 - Camera ID
 - Angle data collections per second
 - ArUco marker length in meters
 - Camera calibration filename
 - Marker detector parameters filename
 - Input video filename
 - Output angle data filename