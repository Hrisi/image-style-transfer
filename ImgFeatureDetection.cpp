#include <ImgFeatureDetection.h>


#define COLOR_THRESHOLD 20
#define NMB_REGIONS 4
#define COLOR_PURITY_THRESHOLD 30
#define MIN_PEAK_DISTANCE 70


void showHistogram(MatND hist, int bins){
    double maxVal = 0;
    minMaxLoc(hist, 0, &maxVal, 0);

    int scale = 4;
    Mat histImg = Mat::zeros(512, 256 * scale, CV_8UC3);

    for(int i = 0; i < bins; i++){
      float binVal = hist.at<float>(i);
      int intensity = cvRound(binVal * 512 / maxVal);
      rectangle(histImg, Point(i*scale, 512),
                Point((i + 1) * scale - 1, 512 - intensity),
                Scalar::all(255),
                CV_FILLED);
    }
    namedWindow("H-S Histogram", 1);
    imshow("H-S Histogram", histImg);
    waitKey(0);
}



void calcHist1D(Mat& array, MatND& hist, int channels, int bins){
  int histChannels[] = {channels};
  int histBins[] = {bins};
  float hueRanges[] = {0, bins};
  const float* ranges[] ={hueRanges};
  calcHist(&array, 1, histChannels, Mat(), hist, 1, histBins, ranges,
           true, false);
}


bool isPeak(std::vector<int>& peaks, int currentPeak, MatND hist, int minSize,
            float minPeakDist, int bins){
  int regionPixelsCnt = hist.at<float>(currentPeak);
  for (int j = 1; j < 10; j++){
    if (currentPeak - j >= 0){
      regionPixelsCnt += hist.at<float>(currentPeak - j);
    }
    if (currentPeak + j < bins){
      regionPixelsCnt += hist.at<float>(currentPeak + j);
    }
  }
  if (regionPixelsCnt < minSize){
    return false;
  }

  for (int i = 0; i < peaks.size(); i++){
    if (currentPeak - peaks[i] < minPeakDist){
      if (hist.at<float>(peaks[i]) <= hist.at<float>(currentPeak)){
        peaks.erase(peaks.begin() + i);
      }
      else{
        return false;
      }
    }
  }
  return true;
}


void findHistPeaks(MatND hist, std::vector<int>& peaks, int histBins,
                   int minSize, float minPeakDist){
  int flagUp = false;

  for (int i = 1; i < histBins; i++){
    if (flagUp && hist.at<float>(i - 1) > hist.at<float>(i)){
      if (isPeak(peaks, i - 1, hist, minSize, minPeakDist, histBins)){
        peaks.push_back(i - 1);
        flagUp = false;
      }
    }
    if (!flagUp && hist.at<float>(i - 1) < hist.at<float>(i)){
      flagUp = true;
    }
    if (i == 1 && hist.at<float>(i - 1) > hist.at<float>(i)){
      if (isPeak(peaks, i - 1, hist, minSize, minPeakDist, histBins)){
        peaks.push_back(i - 1);
      }
    }
  }
}
