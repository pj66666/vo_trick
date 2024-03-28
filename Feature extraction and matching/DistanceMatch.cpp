#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat image1 = imread("./000000.png");    
    Mat image2 = imread("./000001.png");  
    namedWindow("Image 1", 0);
    namedWindow("Image 2", 0);
    imshow("Image 1", image1);
    imshow("Image 2", image2);

    // 提取特征点    
    Ptr<FeatureDetector> surfDetector = ORB::create();  // 使用 ORB 特征检测器
    vector<KeyPoint> keyPoint1, keyPoint2;
    surfDetector->detect(image1, keyPoint1);
    surfDetector->detect(image2, keyPoint2);

    // 计算特征描述子    
    Ptr<DescriptorExtractor> surfDescriptor = ORB::create();  // 使用 ORB 描述子提取器
    Mat imageDesc1, imageDesc2;
    surfDescriptor->compute(image1, keyPoint1, imageDesc1);
    surfDescriptor->compute(image2, keyPoint2, imageDesc2);

    // 特征点匹配    
    BFMatcher matcher;
    vector<DMatch> matchePoints;
    matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
    cout << "Total match points: " << matchePoints.size() << endl;

    // 匹配筛选    
    double min_dist = 1000, max_dist = 0;
    // 寻找最小距离和最大距离
    for (int i = 0; i < imageDesc1.rows; i++)
    {
        double dist = matchePoints[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    // 选择好的匹配点
    vector<DMatch> good_matches;
    for (int i = 0; i < imageDesc1.rows; i++)
    {
        if (matchePoints[i].distance <= max(3 * min_dist, 0.1))
            good_matches.push_back(matchePoints[i]);
    }

    cout << "Total good match points: " << good_matches.size() << endl;

    // 显示匹配结果
    Mat img_match;
    drawMatches(image1, keyPoint1, image2, keyPoint2, good_matches, img_match);
    namedWindow("Distance Match", 0);
    imshow("Distance Match", img_match);
    imwrite("distance_match.jpg", img_match);

    waitKey();
    return 0;
}
