#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ColorSpaces.h>
#include <MongeTransform.h>

using namespace cv;

#define GREY_THRESHOLD 0.3
#define EPS 0.008856
#define K 903.3


float clipImg(float value, float lower, float upper){
  return std::max(lower, std::min(value, upper));
}


float sRGBCompanding(float value){
  if (value <= 0.0031308){
    return value * 12.92;
  }

  return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}


float sRGBInverseCompanding(float value){
  if (value <= 0.04045){
    return value / 12.92;
  }
  
  return pow((value + 0.055)/ 1.055, 2.4);
}


void conversion(const Mat& original, Mat& result, const Mat& matrix,
                int rows, int cols){
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < cols; j++){
      float channel1 =  original.at<Vec3f>(i, j).val[0];
      float channel2 =  original.at<Vec3f>(i, j).val[1];
      float channel3 =  original.at<Vec3f>(i, j).val[2];

      for (int k = 0; k < 3; k++){
        result.at<Vec3f>(i, j).val[k] = matrix.at<float>(k, 0) * channel1 +
                                        matrix.at<float>(k, 1) * channel2 +
                                        matrix.at<float>(k, 2) * channel3;
      }
    }
  }
}


void convertBGRtoLMS(Mat& bgrImg, Mat& lmsImg){
  Mat matrix = (Mat_<float>(3, 3) << 0.3811, 0.5783, 0.0402,
                                     0.1967, 0.7244, 0.0782,
                                     0.0241, 0.1288, 0.8444);

  for (int i = 0; i < bgrImg.rows; i++){
    for (int j = 0; j < bgrImg.cols; j++){
      float r = bgrImg.at<Vec3f>(i, j).val[2];
      float g = bgrImg.at<Vec3f>(i, j).val[1];
      float b = bgrImg.at<Vec3f>(i, j).val[0];

      b = sRGBInverseCompanding(b);
      g = sRGBInverseCompanding(g);
      r = sRGBInverseCompanding(r);

      for (int k = 0; k < 3; k++){
        lmsImg.at<Vec3f>(i, j).val[k] = matrix.at<float>(k, 0) * r +
                                        matrix.at<float>(k, 1) * g +
                                        matrix.at<float>(k, 2) * b;
        //lmsImg.at<Vec3f>(i, j).val[k] =
        //  clipImg(lmsImg.at<Vec3f>(i, j).val[k], 0, 1);
        lmsImg.at<Vec3f>(i, j).val[k] = log10(lmsImg.at<Vec3f>(i, j).val[k]);
      }
    }
  }
}


void convertLMStoLab(const Mat& lmsImg, Mat& labImg){
  Mat base1 = (Mat_<float>(3, 3) << 0.57735026918, 0, 0,
                                    0, 0.40824829046, 0,
                                    0, 0, 0.70710678118);
  Mat base2 = (Mat_<float>(3, 3) << 1, 1, 1,
                                    1, 1, -2,
                                    1, -1, 0);
  Mat matrix = Mat(3, 3, CV_32F);
  dotProduct(base1, base2, matrix);

  conversion(lmsImg, labImg, matrix, lmsImg.rows, lmsImg.cols);
}


void convertLabtoLMS(const Mat& labImg, Mat& lmsImg){
  Mat base1 = (Mat_<float>(3, 3) << 1, 1, 1,
                                    1, 1, -1,
                                    1, -2, 0);
  Mat base2 = (Mat_<float>(3, 3) << 0.57735026918, 0, 0,
                                    0, 0.40824829046, 0,
                                    0, 0, 0.7071067811);
  Mat matrix = Mat(3, 3, CV_32F);
  dotProduct(base1, base2, matrix);
  std::cout << matrix << std::endl;

  conversion(labImg, lmsImg, matrix, lmsImg.rows, lmsImg.cols);

  for (int i = 0; i < lmsImg.rows; i++){
    for (int j = 0; j < lmsImg.cols; j++){
      for (int k = 0; k < 3; k++){
        //lmsImg.at<Vec3f>(i, j).val[k] =
        //  clipImg(lmsImg.at<Vec3f>(i, j).val[k], 0, 1);
        lmsImg.at<Vec3f>(i, j).val[k] = pow(10, lmsImg.at<Vec3f>(i, j).val[k]);
        if (lmsImg.at<Vec3f>(i, j).val[k] < 0){
          std::cout << "!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        }
        //std::cout << lmsImg.at<Vec3f>(i, j).val[k] << std::endl;
      }
    }
  }
}


void convertLMStoBGR(const Mat& lmsImg, Mat& bgrImg){
  Mat matrix = (Mat_<float>(3, 3) << 4.4679, -3.5873, 0.1193,
                                    -1.2186, 2.3809, -0.1624,
                                    0.0497, -0.2439, 1.2045);

  for (int i = 0; i < bgrImg.rows; i++){
    for (int j = 0; j < bgrImg.cols; j++){
      float l = lmsImg.at<Vec3f>(i, j).val[0];
      float m = lmsImg.at<Vec3f>(i, j).val[1];
      float s = lmsImg.at<Vec3f>(i, j).val[2];

      //std::cout << l  << " " << lmsImg.at<Vec3f>(i, j).val[0] << " " << m << " " << s << std::endl;
      for (int k = 0; k < 3; k++){
        bgrImg.at<Vec3f>(i, j).val[2 - k] = matrix.at<float>(k, 0) * l +
                                            matrix.at<float>(k, 1) * m +
                                            matrix.at<float>(k, 2) * s;
        bgrImg.at<Vec3f>(i, j).val[2 - k] =
          sRGBCompanding(bgrImg.at<Vec3f>(i, j).val[2 - k]);
        //std::cout << bgrImg.at<Vec3f>(i, j).val[k] << std::endl;

        //std::cout << bgrImg.at<Vec3f>(i, j).val[2 - k] << std::endl;
        //bgrImg.at<Vec3f>(i, j)[2 - k] =
        //  clipImg(bgrImg.at<Vec3f>(i, j).val[2 - k], 0, 1);
      }
    }
  }
}


void convertBGRtoYUV(const Mat& bgrImg, Mat& yuvImg){
  Mat matrix = (Mat_<float>(3, 3) << 0.299, 0.587, 0.114,
                                     -0.229, -0.587, 0.886,
                                     0.701, -0.587, -0.114);

  for (int i = 0; i < bgrImg.rows; i++){
    for (int j = 0; j < bgrImg.cols; j++){
      float r = bgrImg.at<Vec3f>(i, j).val[2];
      float g = bgrImg.at<Vec3f>(i, j).val[1];
      float b = bgrImg.at<Vec3f>(i, j).val[0];

      for (int k = 0; k < 3; k++){
        yuvImg.at<Vec3f>(i, j).val[k] = matrix.at<float>(k, 0) * r +
                                        matrix.at<float>(k, 1) * g +
                                        matrix.at<float>(k, 2) * b;
      }
    }
  }
}


bool computeBGRGrayWhitePoint(const Mat& bgrImg, Mat& whitePoint,
                              Mat& prevYUVWhitePoint){
  float sumElementsByChannel[2] = {0, 0};
  int cntElm = 0;
  Mat matrix = (Mat_<float>(3, 3) << 0.299, 0.587, 0.114,
                                     -0.229, -0.587, 0.886,
                                     0.701, -0.587, -0.114);

  Mat invMatrix = matrix.inv();
  Mat curYUVWhitePoint = Mat(1, 1, CV_32FC3);

  Mat yuvImg = Mat(bgrImg.rows, bgrImg.cols, CV_32FC3);

  convertBGRtoYUV(bgrImg, yuvImg);

  for (int i = 0; i < yuvImg.rows; i++){
    for (int j = 0; j < yuvImg.cols; j++){
      float y = yuvImg.at<Vec3f>(i, j).val[0];
      float u = yuvImg.at<Vec3f>(i, j).val[1];
      float v = yuvImg.at<Vec3f>(i, j).val[2];
      float ratio = (abs(u) + abs(v)) / y;
      if (ratio < GREY_THRESHOLD){
        sumElementsByChannel[0] += u;
        sumElementsByChannel[1] += v;
        cntElm++;
      }
    }
  }

  curYUVWhitePoint.at<Vec3f>(0, 0).val[0] = 1;
  curYUVWhitePoint.at<Vec3f>(0, 0).val[1] = sumElementsByChannel[0] / cntElm;
  curYUVWhitePoint.at<Vec3f>(0, 0).val[2] = sumElementsByChannel[1] / cntElm;

  // termination criteria
  float curY = curYUVWhitePoint.at<Vec3f>(0, 0).val[0];
  float curU = curYUVWhitePoint.at<Vec3f>(0, 0).val[1];
  float curV = curYUVWhitePoint.at<Vec3f>(0, 0).val[2];
  float prevU = prevYUVWhitePoint.at<Vec3f>(0, 0).val[1];
  float prevV = prevYUVWhitePoint.at<Vec3f>(0, 0).val[2];
  float norm = sqrt(pow(curU - prevU, 2) + pow(curV - prevV, 2));

  if (!std::isnan(prevU) && (norm < pow(10, -6) ||
                             std::max(abs(curU), abs(curV)) < 0.001)){
    return true;
  }

  prevYUVWhitePoint.at<Vec3f>(0, 0).val[0] = curY;
  prevYUVWhitePoint.at<Vec3f>(0, 0).val[1] = curU;
  prevYUVWhitePoint.at<Vec3f>(0, 0).val[2] = curV;
  
  for (int i = 0; i < 3; i++){
    whitePoint.at<Vec3f>(0, 0).val[2 - i] = invMatrix.at<float>(i, 0) * curY +
                                            invMatrix.at<float>(i, 1) * curU +
                                            invMatrix.at<float>(i, 2) * curV;
    whitePoint.at<Vec3f>(0, 0).val[2 - i] =
      clipImg(whitePoint.at<Vec3f>(0, 0).val[2 - i], 0, 1);
  }

  return false;
}


void XYZtoxy(const Mat& XYZImg, Mat& xyImg){
  float sum =
    XYZImg.at<Vec3f>(0, 0).val[0] +
    XYZImg.at<Vec3f>(0, 0).val[1] +
    XYZImg.at<Vec3f>(0, 0).val[2];

  xyImg.at<Vec2f>(0, 0).val[0] = XYZImg.at<Vec3f>(0, 0).val[0] / sum;
  xyImg.at<Vec2f>(0, 0).val[1] = XYZImg.at<Vec3f>(0, 0).val[1] / sum;
}


void xytoXYZ(const Mat& xyImg, float maxLuminance, Mat& XYZImg){
  XYZImg.at<Vec3f>(0, 0).val[0] = 
    xyImg.at<Vec2f>(0, 0).val[0] * maxLuminance / xyImg.at<Vec2f>(0, 0).val[1];
  XYZImg.at<Vec3f>(0, 0).val[1] = maxLuminance;
  XYZImg.at<Vec3f>(0, 0).val[2] = 
    (1 - xyImg.at<Vec2f>(0, 0).val[0] - xyImg.at<Vec2f>(0, 0).val[1]) *
    maxLuminance / xyImg.at<Vec2f>(0, 0).val[1];
}


float product(const Mat& whitePoint, const Mat& invMatrix, int k){
  return invMatrix.at<float>(k, 0) * whitePoint.at<Vec3f>(0, 0).val[0] +
         invMatrix.at<float>(k, 1) * whitePoint.at<Vec3f>(0, 0).val[1] +
         invMatrix.at<float>(k, 2) * whitePoint.at<Vec3f>(0, 0).val[2]; 
}


void getRGBtoXYZMatrix(Mat& whitePoint, Mat& RGBtoXYZMatrix){
  // using the sRGB chromaticity coordinates
  float xR = 0.64;
  float yR = 0.33;
  float xG = 0.3;
  float yG = 0.6;
  float xB = 0.15;
  float yB = 0.06;

  float XR, YR, ZR, XB, YB, ZB, XG, YG, ZG;

  XR = xR / yR;
  YR = 1;
  ZR = (1 - xR - yR) / yR;

  XG = xG / yG;
  YG = 1;
  ZG = (1 - xG - yG) / yG;

  XB = xB / yB;
  YB = 1;
  ZB = (1 - xB - yB) / yB;

  Mat matrix = (Mat_<float>(3, 3) << XR, XG, XB,
                                     YR, YG, YB,
                                     ZR, ZG, ZB);
  Mat invMatrix = matrix.inv();
  
  float sR, sG, sB;
  sR = product(whitePoint, invMatrix, 0);

  sG = product(whitePoint, invMatrix, 1);

  sB = product(whitePoint, invMatrix, 2);

  RGBtoXYZMatrix = (Mat_<float>(3, 3) << sR * XR, sG * XG, sB * XB,
                                         sR * YR, sG * YG, sB * YB,
                                         sR * ZR, sG * ZG, sB * ZB);
}


void convertBGRtoXYZ(Mat& bgrImg, Mat& whitePoint, Mat& xyzImg){
  Mat M = Mat(3, 3, CV_32F);
  getRGBtoXYZMatrix(whitePoint, M);

  for (int i = 0; i < bgrImg.rows; i++){
    for (int j = 0; j < bgrImg.cols; j++){
      for (int k = 0; k < 3; k++){
        bgrImg.at<Vec3f>(i, j).val[k] =
          sRGBInverseCompanding(bgrImg.at<Vec3f>(i, j).val[k]);
      }
      for (int k = 0; k < 3; k++){
        xyzImg.at<Vec3f>(i, j).val[k] = 
          M.at<float>(k, 0) * bgrImg.at<Vec3f>(i, j).val[2] +
          M.at<float>(k, 1) * bgrImg.at<Vec3f>(i, j).val[1] +
          M.at<float>(k, 2) * bgrImg.at<Vec3f>(i, j).val[0];
      }
    }
  }
}


void convertXYZtoBGR(const Mat& xyzImg, Mat& XYZWhitePoint, Mat& bgrImg){
  Mat matXYZtoLab = Mat(3, 3, CV_32F);

  getRGBtoXYZMatrix(XYZWhitePoint, matXYZtoLab);
  matXYZtoLab = matXYZtoLab.inv();

  for (int i = 0; i < xyzImg.rows; i++){
    for (int j = 0; j < xyzImg.cols; j++){
      for (int k = 0; k < 3; k++){
        bgrImg.at<Vec3f>(i, j).val[2 - k] = 
          matXYZtoLab.at<float>(k, 0) * xyzImg.at<Vec3f>(i, j).val[0] +
          matXYZtoLab.at<float>(k, 1) * xyzImg.at<Vec3f>(i, j).val[1] +
          matXYZtoLab.at<float>(k, 2) * xyzImg.at<Vec3f>(i, j).val[2];
        bgrImg.at<Vec3f>(i, j).val[2 - k] =
          sRGBCompanding(bgrImg.at<Vec3f>(i, j).val[2 - k]);
        bgrImg.at<Vec3f>(i, j).val[2 - k] =
          clipImg(bgrImg.at<Vec3f>(i, j).val[2 - k], 0, 1);
      }
    }
  }
}


void adaptXYZToNewWhitePoint(Mat& XYZWhitePoint, const Mat& BGRWhitePoint){
  Mat sRGBChromCoord = (Mat_<float>(3, 3) << 0.6400, 0.3000, 0.1500,
                                             0.3300, 0.6000, 0.0600, 	
                                             0.2126, 0.7152, 0.0722);
  Mat invsRGBChromCoord = sRGBChromCoord.inv();

  Mat XYZD65 = (Mat_<float>(1, 3) << 0.9642, 1.0000, 0.8249);
  Mat xyWhitePoint = Mat(BGRWhitePoint.rows, BGRWhitePoint.cols, CV_32FC3);
  for (int k = 0; k < 3; k++){
    xyWhitePoint.at<Vec3f>(0, 0).val[k] = product(XYZD65, invsRGBChromCoord, k);
  }

  Mat T = (Mat_<float>(3, 3) << xyWhitePoint.at<Vec3f>(0, 0).val[0], 0, 0,
                                0, xyWhitePoint.at<Vec3f>(0, 0).val[1], 0,
                                0, 0, xyWhitePoint.at<Vec3f>(0, 0).val[2]);

  Mat tmpMat = Mat(3, 3, CV_32F);
  dotProduct(sRGBChromCoord, T, tmpMat);
  Mat linBGRWhitePoint = Mat(BGRWhitePoint.rows, BGRWhitePoint.cols, CV_32FC3);

  for (int i = 0; i < 3; i++){
    linBGRWhitePoint.at<Vec3f>(0, 0).val[i] =
      sRGBInverseCompanding(BGRWhitePoint.at<Vec3f>(0, 0).val[i]);
  }
  for (int k = 0; k < 3; k++){
    XYZWhitePoint.at<Vec3f>(0, 0).val[k] =
      linBGRWhitePoint.at<Vec3f>(0, 0).val[2] * tmpMat.at<float>(k, 0) +
      linBGRWhitePoint.at<Vec3f>(0, 0).val[1] * tmpMat.at<float>(k, 1) +
      linBGRWhitePoint.at<Vec3f>(0, 0).val[0] * tmpMat.at<float>(k, 2);
  }
}


void CAT(Mat& bgrImg, Mat& inpWhitePoint, const Mat& tarWhitePoint){
  Mat inpXYZWhitePoint = Mat(inpWhitePoint.rows, inpWhitePoint.cols, CV_32FC3);
  adaptXYZToNewWhitePoint(inpXYZWhitePoint, inpWhitePoint);

  Mat tarXYZWhitePoint = Mat(tarWhitePoint.rows, tarWhitePoint.cols, CV_32FC3);
  adaptXYZToNewWhitePoint(tarXYZWhitePoint, tarWhitePoint);

  Mat xyzImg = Mat(bgrImg.rows, bgrImg.cols, CV_32FC3);
  Mat resXYZImg = Mat(bgrImg.rows, bgrImg.cols, CV_32FC3);
  Mat XYZtoRGBMatrix = Mat(3, 3, CV_32F);

  convertBGRtoXYZ(bgrImg, inpXYZWhitePoint, xyzImg);

  Mat bradford = (Mat_<float>(3, 3) << 0.7328, 0.4296, -0.1624,
                                      -0.7036, 1.6975, 0.0061,
                                      0.0030, -0.0136, 0.9834);

  float roI = product(inpXYZWhitePoint, bradford, 0);
  float gamaI = product(inpXYZWhitePoint, bradford, 1);
  float betaI = product(inpXYZWhitePoint, bradford, 2);

  float roT = product(tarXYZWhitePoint, bradford, 0);
  float gamaT = product(tarXYZWhitePoint, bradford, 1);
  float betaT = product(tarXYZWhitePoint, bradford, 2);

  Mat diagMatrix = (Mat_<float>(3, 3) << roT / roI, 0, 0,
                                         0, gamaT / gamaI, 0,
                                         0, 0, betaT / betaI);

  Mat tmpMatrix = Mat(3, 3, CV_32F);
  dotProduct(bradford.inv(), diagMatrix, tmpMatrix);
  Mat M = Mat(3, 3, CV_32F);
  dotProduct(tmpMatrix, bradford, M);

  for (int i = 0; i < bgrImg.rows; i++){
    for (int j = 0; j < bgrImg.cols; j++){
      for (int k = 0; k < 3; k++){
        resXYZImg.at<Vec3f>(i, j).val[k] = 
          M.at<float>(k, 0) * xyzImg.at<Vec3f>(i, j).val[0] +
          M.at<float>(k, 1) * xyzImg.at<Vec3f>(i, j).val[1] +
          M.at<float>(k, 2) * xyzImg.at<Vec3f>(i, j).val[2];
        resXYZImg.at<Vec3f>(i, j).val[k] =
          clipImg(resXYZImg.at<Vec3f>(i, j).val[k], 0, 1);
      }
    }
  }

  convertXYZtoBGR(resXYZImg, inpXYZWhitePoint, bgrImg);
}


void convertBGRToLab(Mat& bgrImg, Mat& BGRWhitePoint, Mat& labImg){
  Mat xyzImg = Mat(bgrImg.rows, bgrImg.cols, CV_32FC3);
  Mat XYZWhitePoint = Mat(1, 1, CV_32FC3);
  adaptXYZToNewWhitePoint(XYZWhitePoint, BGRWhitePoint);
  convertBGRtoXYZ(bgrImg, XYZWhitePoint, xyzImg);

  for (int i = 0; i < xyzImg.rows; i++){
    for (int j = 0; j < xyzImg.cols; j++){
      float xr = xyzImg.at<Vec3f>(i, j).val[0] /
                 XYZWhitePoint.at<Vec3f>(0, 0).val[0];
      float yr = xyzImg.at<Vec3f>(i, j).val[1] /
                 XYZWhitePoint.at<Vec3f>(0, 0).val[1];
      float zr = xyzImg.at<Vec3f>(i, j).val[2] /
                 XYZWhitePoint.at<Vec3f>(0, 0).val[2];
      float fx = (xr > EPS) ? pow(xr, 1 / 3.0) : (K * xr + 16) / 116;
      float fy = (yr > EPS) ? pow(yr, 1 / 3.0) : (K * yr + 16) / 116;
      float fz = (zr > EPS) ? pow(zr, 1 / 3.0) : (K * zr + 16) / 116;

      labImg.at<Vec3f>(i, j).val[0] = 116 * fy - 16;
      labImg.at<Vec3f>(i, j).val[1] = 500 * (fx - fy);
      labImg.at<Vec3f>(i, j).val[2] = 200 * (fy - fz);
    }
  }
}


void convertLabToBGR(const Mat& labImg, const Mat& BGRWhitePoint, Mat& bgrImg){
  std::cout << labImg << std::endl;
  Mat xyzImg = Mat(labImg.rows, labImg.cols, CV_32FC3);
  Mat XYZWhitePoint = Mat(1, 1, CV_32FC3);
  adaptXYZToNewWhitePoint(XYZWhitePoint, BGRWhitePoint);

  for (int i = 0; i < labImg.rows; i++){
    for (int j = 0; j < labImg.cols; j++){
      float L = labImg.at<Vec3f>(i, j).val[0];
      float a = labImg.at<Vec3f>(i, j).val[1];
      float b = labImg.at<Vec3f>(i, j).val[2];

      float fy = (L + 16) / 116;
      float fx = a / 500 + fy;
      float fz = fy - b / 200;

      float xr = (pow(fx, 3) > EPS) ? pow(fx, 3) : (116 * fx - 16) / K;
      float yr = (L > K * EPS) ? pow((L + 16) / 116, 3) : L / K;
      float zr = (pow(fz, 3) > EPS) ? pow(fz, 3) : (116 * fz - 16) / K;

      xyzImg.at<Vec3f>(i, j).val[0] = xr * XYZWhitePoint.at<Vec3f>(i, j).val[0];
      xyzImg.at<Vec3f>(i, j).val[1] = yr * XYZWhitePoint.at<Vec3f>(i, j).val[1];
      xyzImg.at<Vec3f>(i, j).val[2] = zr * XYZWhitePoint.at<Vec3f>(i, j).val[2];
    }
  }

  convertXYZtoBGR(xyzImg, XYZWhitePoint, bgrImg);
}
