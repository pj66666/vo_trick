#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "ORBextractor.h"
#include <cmath>


using namespace std;
void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
/*
 *  @brief 主函数     
 *  @param[in] argc    参数个数
 *  @param[in] argv     argv[0]：可执行文件   argv[1]：数据集路径
*/
int main(int argc, char **argv)
{
    
    ORBextractor* mpORBextractor = new ORBextractor(1000,8,20,7,1.2);
    std::vector<cv::KeyPoint>  mvKeypoints1,mvKeypoints2;
    cv::Mat mdescriptors1,mdescriptors2;

    cv::Mat img1 = cv::imread("/home/pj/pj/trick of VO/ORB_match_demo/000000.png", 0);
    cv::Mat img2 = cv::imread("/home/pj/pj/trick of VO/ORB_match_demo/000001.png", 0);

    auto start_ORB = std::chrono::high_resolution_clock::now();
    (*mpORBextractor)(img1, mvKeypoints1,mdescriptors1);
    auto end_ORB = std::chrono::high_resolution_clock::now();
    auto duration_ORB = std::chrono::duration_cast<std::chrono::milliseconds>(end_ORB - start_ORB).count();
    std::cout << "ORB Execution time: " << duration_ORB << " milliseconds" << std::endl; 

    (*mpORBextractor)(img2, mvKeypoints2,mdescriptors2);

    mdescriptors1.convertTo(mdescriptors1, CV_32F);
    mdescriptors2.convertTo(mdescriptors2, CV_32F);
    
    
    auto start = std::chrono::high_resolution_clock::now();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    // Match descriptors
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(mdescriptors1, mdescriptors2, knn_matches, 2);    // 为每个特征点返回至少两个可能匹配的结果

    // Apply ratio test to select good matches
    const float ratio_thresh = 0.8f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "matches Execution time: " << duration << " milliseconds" << std::endl; 

    std::vector<int> vMatches12(mvKeypoints1.size(),-1);
    for(size_t i =0; i < good_matches.size(); i++)
    {
        if (good_matches[i].queryIdx >= 0 && good_matches[i].queryIdx < mvKeypoints1.size())
        {
            vMatches12[good_matches[i].queryIdx] = good_matches[i].trainIdx;
        }
    }


    // 创建一个数组，每个元素都是vector<int>
    int HISTO_LENGTH = 30;
    std::vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    //! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码   
    const float factor = HISTO_LENGTH/360.0f;




    // Step 5 计算匹配点旋转角度差所在的直方图
    for(size_t j = 0; j < vMatches12.size(); j++)
    {
        if(vMatches12[j] == -1)
        {
            continue;
        }
        // 计算匹配特征点的角度差，这里单位是角度°，不是弧度
        float rot = mvKeypoints1[j].angle-mvKeypoints2[vMatches12[j]].angle;
        if(rot<0.0)
        {
            rot+=360.0f;
        }

        // 前面factor = HISTO_LENGTH/360.0f 
        // bin = rot / 360.of * HISTO_LENGTH 表示当前rot被分配在第几个直方图bin  
        int bin = round(rot*factor);
        // 如果bin 满了又是一个轮回
        if(bin==HISTO_LENGTH)
        {
            bin=0;                
        }

        assert(bin>=0 && bin<HISTO_LENGTH);
        // 把每一个特征点加到对应的直方图区间
        rotHist[bin].push_back(j);
    }

    
// Step 6 筛除旋转直方图中次要部分


    int ind1=-1;
    int ind2=-1;
    int ind3=-1;
    // 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
    ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

    int s =  good_matches.size();
    for(int i=0; i<HISTO_LENGTH; i++)
    {
        if(i==ind1 || i==ind2 || i==ind3)
            continue;
        // 剔除掉不在前三的匹配对，因为他们不符合“主流旋转方向”    
        for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
        {
            int idx1 = rotHist[i][j];
            if(vMatches12[idx1]>=0)
            {
                vMatches12[idx1]=-1;
                s--;
            }
        }
    }
    
    std::cout << "每一帧提取到的特征点总数:" << mvKeypoints1.size() << std::endl;
    std::cout << "旋转直方图筛选之前(FLANN)的匹配数：" <<  good_matches.size() << std::endl;
    std::cout << "旋转直方图筛选之后的匹配数：" <<  s << std::endl;





    // Draw the matches
    cv::Mat img_matches;
    cv::drawMatches(img1, mvKeypoints1, img2, mvKeypoints2, good_matches, img_matches);

    cv::Mat img1_color, img2_color;
    cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);

    // 创建新的图像用于绘制匹配结果和线
    cv::Mat img_matches_with_lines(img1_color.rows, img1_color.cols + img2_color.cols, CV_8UC3);
    // 将两张图像拼接到一个大图像上
    cv::hconcat(img1_color, img2_color, img_matches_with_lines);

    // Draw lines between matched keypoints based on filtered matches (vMatches12)
    for (size_t i = 0; i < vMatches12.size(); i++) {
        if (vMatches12[i] != -1) {
            cv::Point2f pt1 = mvKeypoints1[i].pt;
            cv::Point2f pt2 = mvKeypoints2[vMatches12[i]].pt + cv::Point2f(img1.cols, 0); // 加上图像2的偏移

            // 生成随机颜色
            cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);

            // 画线
            cv::line(img_matches_with_lines, pt1, pt2, color, 1); // Draw a random color line

            // 画点
            cv::circle(img_matches_with_lines, pt1, 4, color);  // -1 表示填充圆心
            cv::circle(img_matches_with_lines, pt2, 4, color);  // -1 表示填充圆心
        }
    }

    // Display the matches
    cv::imshow("Matches", img_matches);
    cv::imshow("rotH", img_matches_with_lines);
    cv::imwrite("knn_match.jpg", img_matches);
    cv::imwrite("rotH.jpg", img_matches_with_lines);
    cv::waitKey(0);

    return 0;
}

void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}