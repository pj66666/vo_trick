#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>  

using namespace cv;
using namespace std;

int main()
{
    Mat grayImage1 = imread("./000000.png");    
    Mat grayImage2 = imread("./000001.png");   
    namedWindow("Image 1", 0);
    namedWindow("Image 2", 0);
    imshow("Image 1", grayImage1);
    imshow("Image 2", grayImage2);


    // 提取特征点和描述子    
    Ptr<FeatureDetector> orbDetector = ORB::create(); // 使用 ORB 特征检测器
    vector<KeyPoint> keyPoint1, keyPoint2;
    orbDetector->detect(grayImage1, keyPoint1);
    orbDetector->detect(grayImage2, keyPoint2);

    Ptr<DescriptorExtractor> orbDescriptor = ORB::create(); // 使用 ORB 描述子提取器
    Mat descriptor1, descriptor2;
    orbDescriptor->compute(grayImage1, keyPoint1, descriptor1);
    orbDescriptor->compute(grayImage2, keyPoint2, descriptor2);

    // 特征点匹配    
    BFMatcher matcher(NORM_HAMMING); // 使用汉明距离
    vector<vector<DMatch>> knnMatches;
    const int k = 2;
    matcher.knnMatch(descriptor1, descriptor2, knnMatches, k);

    vector<DMatch> matches;
    const float minRatio = 0.6;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        const DMatch& bestMatch = knnMatches[i][0];
        const DMatch& betterMatch = knnMatches[i][1];
        float distanceRatio = bestMatch.distance / betterMatch.distance;
        if (distanceRatio < minRatio)
            matches.push_back(bestMatch);
    }

    cout << "Total match points: " << matches.size() << endl;

    // 显示匹配结果
    Mat img_match;
    drawMatches(grayImage1, keyPoint1, grayImage2, keyPoint2, matches, img_match);
    namedWindow("knn_match", 0);
    imshow("knn_match", img_match);
    imwrite("knn_match.jpg", img_match);

    waitKey();
    return 0;
}
