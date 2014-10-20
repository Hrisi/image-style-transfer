#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ReinhardTransform.h>


float applyReinhard(float value, float inpMean, float tarMean, float inpCov,
                    float tarCov){
  double newValue;

  newValue = tarCov * (value - inpMean) / inpCov + tarMean;

  return newValue;
}
