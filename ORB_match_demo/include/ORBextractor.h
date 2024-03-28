#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H
#include <vector>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


/**
 * @brief 将提取器节点分成4个子节点，同时也完成图像区域的划分、特征点归属的划分，以及相关标志位的置位
 * @param[in & out] n1  提取器节点1：左上
 * @param[in & out] n2  提取器节点1：右上
 * @param[in & out] n3  提取器节点1：左下
 * @param[in & out] n4  提取器节点1：右下
 */
class ExtractorNode  // 定义划分节点类
{
public:
    ExtractorNode():bNoMore(false){}  // 构造函数，初始化列表参数
    //
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);    // 分配节点函数

    std::vector<cv::KeyPoint> vKeys; // 当前母节点区域内关键点数组vector
    cv::Point2i UL, UR, BL, BR;      // 一个矩形图像区域四个角的坐标(当前节点-母节点)
    std::list<ExtractorNode>::iterator lit;     // 构建list容器
    bool bNoMore;   // 表示属于当前节点的特征点数量=1则为true
};



class ORBextractor
{
public:
    // 声明时候可以不写初始化列表
    ORBextractor( int nfeatures,int nlevels,int initialThFast,int minThFast,float scaleFactor);

    ~ORBextractor();
  
    void ComputePyramid(cv::Mat image);

    float IC_Angle(const cv::Mat& image, cv::Point2f pt,  const std::vector<int> & u_max);

    void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& umax);

    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);

    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level);

    void computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                               const std::vector<cv::Point>& pattern);

    void computeOrbDescriptor(const cv::KeyPoint& kpt,
                                 const cv::Mat& img, const cv::Point* pattern,
                                 uchar* desc);                               

    void operator()( cv::InputArray image,std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    inline std::vector<cv::Mat>  GetmvImagePyramid(){
        return mvImagePyramid;
    }

    inline  std::vector<float> GetscaleFactor(){
        return mvscaleFactor;
    }

    inline std::vector<float>  GetInvscaleFactor(){
        return mvInvscaleFactor;
    }

    inline std::vector<cv::Point>  Getpattern(){
        return pattern;
    }


private:

        
    int nfeatures;      // 总的特征点数2000
    int nlevels;            // 8

    int initialThFast;  // 初始的FAST响应值阈值     20   如果检测不到，就要降低阈值
    int minThFast;      //  最小的FAST响应值阈值    7


    float scaleFactor;     // 1.2
    std::vector<float> mvscaleFactor;   // 1.2   1.2*1.2   ......
    std::vector<float> mvInvscaleFactor;   // 1/1.2   1/(1.2*1.2)   ......

    std::vector<cv::Mat> mvImagePyramid;
    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;   
    std::vector<cv::Point> pattern;      // 随机点对--计算描述子

};

#endif // ORBEXTRACTOR_H