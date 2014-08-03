// Place for license

#include <iostream>
#include <opencv2/opencv.hpp>

#include<HistogramMatching.h>

using namespace cv;

typedef std::vector<Mat> ChannelsType;


int main(int argc, char** argv){
  // 4 arguments: file name, input image, target image, result image
  if (argc != 4 ){
    return -1;
  }

  Mat inpImg, tarImg, finalImg, inpImgLab, tarImgLab;

  inpImg = imread(argv[1], 1);
  tarImg = imread(argv[2], 1);

  if (!inpImg.data || !tarImg.data){
    std::cout << "No image data \n";
    return -1;
  }
  
  cvtColor(inpImg, inpImgLab, CV_BGR2Lab, 0);
  cvtColor(tarImg, tarImgLab, CV_BGR2Lab, 0);

  // --- histogram matching
  performMatching(inpImgLab, tarImgLab, 0);
  // histogram matching ---

  cvtColor(inpImgLab, finalImg, CV_Lab2BGR, 0);

  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", finalImg);

  waitKey(0);

  return 0;
}
