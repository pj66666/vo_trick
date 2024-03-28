#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // 读取图像
    Mat image1 = imread("./000000.png");    
    Mat image2 = imread("./000001.png");    
    namedWindow("Image 1", 0);
    namedWindow("Image 2", 0);
    imshow("Image 1", image1);
    imshow("Image 2", image2);


    // 提取特征点    
    Ptr<FeatureDetector> detector = ORB::create();  // 使用 ORB 特征检测器
    vector<KeyPoint> keyPoint1, keyPoint2;
    detector->detect(image1, keyPoint1);
    detector->detect(image2, keyPoint2);

    // 提取特征描述子    
    Ptr<DescriptorExtractor> descriptor = ORB::create();  // 使用 ORB 描述子提取器
    Mat imageDesc1, imageDesc2;
    descriptor->compute(image1, keyPoint1, imageDesc1);
    descriptor->compute(image2, keyPoint2, imageDesc2);

    // 实现特征点匹配
    BFMatcher matcher;
    vector<DMatch> matches;
    // 进行匹配
    matcher.match(imageDesc1, imageDesc2, matches);

    // 设置最小匹配点数
    const int minNumbermatchesAllowed = 8;
    if (matches.size() < minNumbermatchesAllowed)
        return 0;

    // 为了计算变换矩阵
    vector<Point2f> srcPoints(matches.size());
    vector<Point2f> dstPoints(matches.size());

    for (size_t i = 0; i < matches.size(); i++) {
        srcPoints[i] = keyPoint2[matches[i].trainIdx].pt;
        dstPoints[i] = keyPoint1[matches[i].queryIdx].pt;
    }
    // 计算变换矩阵
    vector<uchar> inliersMask(srcPoints.size());
    Mat homography;
    homography = findHomography(srcPoints, dstPoints, RANSAC, 5, inliersMask);

    vector<DMatch> inliers;
    for (size_t i = 0; i < inliersMask.size(); i++){
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }
    // 将RANSAC得到的正确匹配点转换为匹配列表
    matches.swap(inliers);
    
    cout << "Total match points: " << matches.size() << endl;

    // 显示匹配结果
    Mat img_match;
    drawMatches(image1, keyPoint1, image2, keyPoint2, matches, img_match);
    namedWindow("RANSAC Match", 0);
    imshow("RANSAC Match", img_match);
    imwrite("ransac_match.jpg", img_match);

    waitKey();
    return 0;
}
