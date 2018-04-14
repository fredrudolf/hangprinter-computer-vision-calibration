/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

namespace
{
const char *about = "Basic marker detection";
const char *keys =
    "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
    "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
    "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
    "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
    "{c        |       | Camera intrinsic parameters. Needed for camera pose }"
    "{l        | 0.1   | Marker side lenght (in meters). Needed for correct scale in camera pose }"
    "{dp       |       | File of marker detector parameters }"
    "{r        |       | show rejected candidates too }";
}

/**
 */
static bool readCameraParameters(string file, Mat &camMat, Mat &distCoeffs, Size *imgSize)
{
   FileStorage fs(file, FileStorage::READ);
   if (!fs.isOpened())
      return false;
   fs["camera_matrix"] >> camMat;
   fs["distortion_coefficients"] >> distCoeffs;
   fs["image_width"] >> (*imgSize).width;
   fs["image_height"] >> (*imgSize).height;
   return true;
}

/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params)
{
   FileStorage fs(filename, FileStorage::READ);
   if (!fs.isOpened())
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

/**
 */
int main(int argc, char *argv[])
{
   CommandLineParser parser(argc, argv, keys);
   parser.about(about);

   if (argc < 2)
   {
      parser.printMessage();
      return 0;
   }

   int dictionaryId = parser.get<int>("d");
   bool showRejected = parser.has("r");
   bool estimatePose = parser.has("c");
   float markerLength = parser.get<float>("l");

   Ptr<aruco::DetectorParameters> detectParams = aruco::DetectorParameters::create();
   if (parser.has("dp"))
   {
      bool readOk = readDetectorParameters(parser.get<string>("dp"), detectParams);
      if (!readOk)
      {
         cerr << "Invalid detector parameters file" << endl;
         return 0;
      }
   }

   if (!parser.check())
   {
      parser.printErrors();
      return 0;
   }

   Ptr<aruco::Dictionary> dictionary =
       aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

   Mat camMat, distCoeffs, camMatRemap, map1, map2;
   Size imgSize;
   if (estimatePose)
   {
      string filename = parser.get<string>("c");
      bool readOk = readCameraParameters(filename, camMat, distCoeffs, &imgSize);
      if (!readOk)
      {
         cerr << "Invalid camera file" << endl;
         return 0;
      }
   }
   else
   {
      cerr << "Missing camera file (-c parameter)" << endl;
      return 0;
   }

   // Remap fisheye
   fisheye::estimateNewCameraMatrixForUndistortRectify(camMat, distCoeffs, imgSize, Matx33d::eye(), camMatRemap, 1);
   fisheye::initUndistortRectifyMap(camMat, distCoeffs, Matx33d::eye(), camMatRemap, imgSize, CV_16SC2, map1, map2);

   namedWindow("Frame", WINDOW_NORMAL);
   resizeWindow("Frame", 1024, 768);
    cout << "\"id\",\"tx\",\"ty\",\"tz\",\"rx\",\"ry\",\"rz\"" << endl;
   for (int img_i = 1; img_i < argc; img_i++)
   {
      if (argv[img_i][0] == '-')
         continue;

      Mat img, imgRemap;
      vector<int> ids;
      vector<vector<Point2f>> corners, rejected;
      vector<Vec3d> rvecs, tvecs;
      int id0 = -1;
      Mat aruco_points;

      // Read image
      img = imread(argv[img_i], CV_LOAD_IMAGE_COLOR);
      if (img.size() != imgSize)
      {
         cout << argv[img_i] << " has wrong image size." << endl;
         return 0;
      }
      // Remap from fishey
      remap(img, imgRemap, map1, map2, INTER_LINEAR, BORDER_CONSTANT);

      // detect markers and estimate pose
      aruco::detectMarkers(imgRemap, dictionary, corners, ids, detectParams, rejected);
      if (ids.size() > 0)
      {
         aruco::estimatePoseSingleMarkers(corners, markerLength, camMat, distCoeffs, rvecs, tvecs);
         aruco::drawDetectedMarkers(imgRemap, corners, ids);
         for (unsigned int i = 0; i < ids.size(); i++)
         {
            aruco::drawAxis(imgRemap, camMat, distCoeffs, rvecs[i], tvecs[i], markerLength * 0.5f);
            // Find id0
            cout << ids[i] << ","
                 << tvecs[i][0] << ","
                 << tvecs[i][1] << ","
                 << tvecs[i][2] << ","
                 << rvecs[i][0] << ","
                 << rvecs[i][1] << ","
                 << rvecs[i][2] << endl;
         }
      }

      // Show image
      imshow("Frame", imgRemap);
      char key = (char)waitKey();

      if (key == 27)
         break;
   }

   return 0;
}
