#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat image01 = imread("./000000.png");
    Mat image02 = imread("./000001.png");
    namedWindow("Image 1", 0);
    namedWindow("Image 2", 0);
    imshow("Image 1", image01);
    imshow("Image 2", image02);

    // 提取特征点
    Ptr<FeatureDetector> detector = ORB::create();
    vector<KeyPoint> keyPoint1, keyPoint2;
    detector->detect(image01, keyPoint1);
    detector->detect(image02, keyPoint2);

    // 计算描述子
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Mat descriptor1, descriptor2;
    descriptor->compute(image01, keyPoint1, descriptor1);
    descriptor->compute(image02, keyPoint2, descriptor2);

    // 特征点匹配
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);

    cout << "Total match points: " << matches.size() << endl;

    // 显示匹配结果
    Mat imgMatch;
    drawMatches(image01, keyPoint1, image02, keyPoint2, matches, imgMatch);
    namedWindow("Match", 0);
    imshow("BFmatch", imgMatch);
    imwrite("BFmatch.jpg", imgMatch);

    waitKey();
    return 0;
}
