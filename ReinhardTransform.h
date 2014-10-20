#ifndef __reinhard_transform_h__

#define __reinhard_transform_h__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;


float applyReinhard(float value, float inpMean, float tarMean, float inpCov,
                    float tarCov);


#endif
