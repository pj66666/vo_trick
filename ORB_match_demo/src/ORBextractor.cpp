#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include<iostream>
#include "ORBextractor.h"
#include <chrono>
#include <cmath>

using namespace cv;
using namespace std;

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;
const float factorPI = (float)(CV_PI/180.f);


static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};






/*
 *  @brief      ORB特征提取构造函数—改善Fast角点两大缺点 
                        1 初始化尺度不变性参数mvInvscaleFactor    每层金字塔要提取的特征点数mnFeaturesPerLevel
                        2 初始化旋转不变性(灰度质心法)参数umax
                        3 初始化描述子容器
 *  @param[in] pre_nfeatures    总的特征点数量
 *  @param[in] pre_nlevels    金字塔总层数
 *  @param[in] ppre_initialThFast    Fast角点初始化阈值
 *  @param[in] pre_minThFast    Fast角点最低化阈值
 *  @param[in] pre_scaleFactor    金字塔层尺度
*/
ORBextractor::ORBextractor(int pre_nfeatures,int pre_nlevels,int pre_initialThFast,int pre_minThFast,
            float pre_scaleFactor):nfeatures(pre_nfeatures),nlevels(pre_nlevels),initialThFast(pre_initialThFast),
            minThFast(pre_minThFast),scaleFactor(pre_scaleFactor)
{
    // 1 初始化尺度不变性参数 mvInvscaleFactor

    // 1.1  预分配空间  
    mvscaleFactor.resize(nlevels);             
    mvInvscaleFactor.resize(nlevels);              // float
    mvImagePyramid.resize(nlevels);            // cv::Mat 存储金字塔层变量  每一个元素都代表一个金字塔层
    mnFeaturesPerLevel.resize(nlevels);     // int分配到每层图像中，要提取的特征点数目


    // 1.2 计算尺度因子
    for(int i = 0;i < nlevels; i++)
    {
        mvscaleFactor[i] = pow(scaleFactor, i);            
        mvInvscaleFactor[i] = pow(scaleFactor, -i);
    }

    // 1.3 计算每层要提取特征点数
    // 第n层
    float nDesiredFeaturesPerScale = nfeatures*(1 - mvInvscaleFactor[1])/(1 - (float)pow((double)mvInvscaleFactor[1], (double)nlevels));  

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )    // 注意这里只是计算了前面nlevels-1层，最后一层没有计算，因为取整问题
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);      // 四舍五入
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= mvInvscaleFactor[1];
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);  // 最顶层金字塔图像特征点数


    // 2 初始化旋转不变性(灰度质心法)参数umax
    // 计算化灰度质心法中圆的每一行最大半径  v是行索引，从0-R之间每一行最大的半径 
    umax.resize(HALF_PATCH_SIZE + 1);   //  std::vector<int> umax; 
    // 1. 计算0-vmax
    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1); // 向下取整
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);  // 向上取整
    // 除HALF_PATCH_SIZE * sqrt(2.f) / 2是整数外，vmin=vmax
    const double R_2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;  // R^2
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(R_2 - v * v));   // 四舍五入取整
    // 2.计算vmin(=vmax)-R
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    // 3 计算描述子随机点对std::vector<cv::Point> pattern;    
    const int npoints = 512; // 256点对，即512个点,每个点(x,y)

    const  cv::Point* _pattern = (const cv::Point*)bit_pattern_31_; //  int bit_pattern_31_[256*4]; 
    //pattern.clear(); // 清空 pattern 向量
    // 强制把整型数组指针bit_pattern_31_(数组首地址)转换为cv::Point型，然后赋给Point型指针_pattern
    // std::copy参数_pattern, _pattern + npoints是一个range迭代器，返回[first，last),即*_pattern, *(_pattern + npoints-1)
    std::copy(_pattern, _pattern + npoints, std::back_inserter(pattern));   // std::back_inserter 是一个迭代器适配器，用于在容器的末尾插入元素

}


ORBextractor::~ORBextractor()
{
    std::cout << "特征检测模块结束"<< std::endl;
}


/*
 *  @brief      构建图像金字塔，每层按比例缩放，且扩展EDGE_THRESHOLD—为了提取边缘的ORB特征点
 *  @param[in] image    原始图像
*/
void ORBextractor::ComputePyramid(cv::Mat image)
{

    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvscaleFactor[level];  
        // Size对象sz  Size 是 OpenCV 中的一个类，用于表示图像的尺寸（宽度和高度）
        
        // 1. 计算原图像比例缩放
        cv::Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        
        // 2. 定义temp图像，它的大小为 sz，类型与输入图像 image 相同       Mat构造函数 cv::Mat(Size size, int type);
        cv::Mat temp(sz, image.type());

        mvImagePyramid[level] = temp;

        // Compute the resized image
        if( level != 0 )
        {
            cv::resize(image, // 输入图像
                mvImagePyramid[level],  // 输出图像
                sz,                 // 输出图像的尺寸
                0,  // 水平方向上的缩放系数，留0表示自动计算
                0,  // 水平方向上的缩放系数，留0表示自动计算
                cv::INTER_LINEAR); // 图像缩放的差值算法类型，这里的是双线性插值算法          
            // 5. 扩充图像插值
            cv::copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);      
                           // BORDER_REFLECT_101 表示使用镜像反射方式进行边界扩展。边界上的像素值通过沿边界反射方式获得，即像素值从边界开始反射，例如：abc|d|cba。

                           // BORDER_ISOLATED 表示边界扩展时不复制原始图像的边界值，而是将边界外的像素置为0。      
        }
        else
        {
            cv::copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           cv::BORDER_REFLECT_101);            
        }
        mvImagePyramid[level] = temp;
    }
}


/*
 *  @brief     计算特征点主方向
 *  @param[in] image    总的特征点数量
 *  @param[in] pt    特征点
 *  @param[in & out] u_max  灰度质心法每层搜索列长度u
 * @return cv::fastAtan2((float)m_01, (float)m_10)   特征点主方向
*/
float ORBextractor::IC_Angle(const cv::Mat& image, cv::Point2f pt,  const std::vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;
	
    // cvRound表示取整，以关键点像素地址为center指针   uchar对应灰度图
    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));  

    // 1.单独求圆形区域水平坐标轴这一行像素灰度，这一行y=0，所以只需要计算m10=x*I(x,y)
    // 	 HALF_PATCH_SIZE = 15 = 圆半径 奇数 = 2*数+1
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // 2.对称计算每一行的m10和m01，m10=x(I(X,Y)+I(x,-y))   m01=y(I(X,Y)-I(x,-y))
    int step = (int)image.step1();  // 图像一行的字节数
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v) // 计算15次(行row)
    {
        int v_sum = 0;
        int d = u_max[v];	// 得到这一行的区域最大x坐标范围d(半径)
        for (int u = -d; u <= d; ++u)
        {
            //3. 利用指针计算像素强度I(x,y)  以pt.x,pt.y为中心，列坐标+行数*行字节数 非常巧妙
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);  
            m_10 += u * (val_plus + val_minus);  
        }
        m_01 += v * v_sum;
    }

    return cv::fastAtan2((float)m_01, (float)m_10);  // arctan
}

/*
 *  @brief      调用IC_Angle来计算特征点主方向
 *  @param[in] image    图像层
 *  @param[in] keypoints    特征点
 *  @param[in & out] u_max  灰度质心法每层搜索列长度u
*/
void ORBextractor::computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& umax)
{
    for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);  // 每个关键点主方向
    }
}


/*
 *  @brief      利用四叉树来计算金字塔中图像层的每一个特征点
                        1. 划分图像区域，遍历每一个30*30图像区域，在图像块内寻找Fast角点.这样保证了大部分区域都有点，而不是集中在某一区域
                        2  调用DistributeOctTree,在图像中均匀的选取特征点
                        3  记录每一层筛选后的所有特征点，然后求取主方向
 *  @param[in] allKeypoints    外层vector即对应图像金字塔的每一层，里层对应图像层的所有特征点
*/
void ORBextractor::ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints)
{
    
    allKeypoints.resize(nlevels);  

    const float W = 30; // 图像网格块cell的尺寸，是个正方形

    for (int level = 0; level < nlevels; ++level)
    {   
        // 1. 计算边界
        // 这里的3是因为在计算FAST特征点的时候，需要建立一个半径为3的圆
        const int minBorderX = EDGE_THRESHOLD-3;  // 16  
        const int minBorderY = minBorderX; // 16 
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        // 存储需要进行平均分配的特征点
        std::vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures*10); 

        // 计算进行特征点提取的图像区域尺寸
        const float width = (maxBorderX-minBorderX);
        const float height = (maxBorderY-minBorderY);

        const int nCols = width/W;  // 列方向有多少图像网格块(30*30)
        const int nRows = height/W; // 行方向有多少图像网格块(30*30)
        // 计算每个图像网格块所占的像素行数和列数
        const int wCell = ceil(width/nCols);
        const int hCell = ceil(height/nRows);

        // 2. 开始遍历图像网格(30*30)，以行开始遍历的(从第一列的行)
        for(int i=0; i<nRows; i++)
        {
            // 行方向第i个图像网格块
            // 计算当前网格最大的行坐标，这里的+6=+3+3，即考虑到了多出来3是为了cell边界像素进行FAST特征点提取用
            // 前面的EDGE_THRESHOLD指的应该是提取后的特征点所在的边界，所以minBorderY是考虑了计算半径时候的图像边界
            const float iniY =minBorderY+i*hCell;
            float maxY = iniY+hCell+6;

            if(iniY>=maxBorderY-3)
                continue;
            if(maxY>maxBorderY)
                maxY = maxBorderY;

            // 3. 开始列的遍历
            for(int j=0; j<nCols; j++)
            {
                const float iniX =minBorderX+j*wCell;
                float maxX = iniX+wCell+6;
                if(iniX>=maxBorderX-6)
                    continue;
                if(maxX>maxBorderX)
                    maxX = maxBorderX;

                // 4. 提取FAST点, 自适应阈值
                // 这个向量存储这个cell中的特征点
                std::vector<cv::KeyPoint> vKeysCell;
                cv::FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), // 图像块区域
                    // vKeysCell存储角点  initialThFast初始阈值
                    vKeysCell,initialThFast,true); // true表示使能非极大值抑制，避免角过于集中

                // 5. 如果这个图像块中使用默认的FAST检测阈值没有能够检测到角点
                if(vKeysCell.empty())
                {
                    // 使用更低的阈值minThFAST来进行重新检测
                    cv::FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,minThFast,true); // true表示使能非极大值抑制，避免特征点过于集中
                }

                if(!vKeysCell.empty())
                {  // 6. 遍历当前层中的所有FAST角点，恢复其在整个金字塔当前层图像下的坐标
                   // 因为这里提取的特征点的坐标只是小方块30*30的坐标，恢复特征点在当前金字塔层的坐标
                    for(std::vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }

            }
        }

        //  声明一个对当前图层的特征点的容器的引用!!!!!!!!!!!!      这个引用很漂亮
        std::vector<cv::KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        // 利用四叉树均匀选取当前金字塔层特征点。   vToDistributeKeys是当前层提取的角点
        // mnFeaturesPerLevel[level]是当前层分配的特征点数目   level当前层级
        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

        // PATCH_SIZE是对于底层的初始图像来说的，现在要根据当前图层的尺度缩放倍数进行缩放得到缩放后的PATCH大小 
        // 估计是图像越小，所占比例越大，所以这里用的是mvscaleFactor而不是mvInvscaleFactor
        const int scaledPatchSize = PATCH_SIZE*mvscaleFactor[level];

        // 获取剔除过程后保留下来的特征点数目，因为每个节点内只保留一个特征点
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            // 对每一个保留下来的特征点，恢复到相对于当前图层“边缘扩充图像下”的坐标系的坐标
            // 因为现在是在排除上下左右都为16的区域进行的特征提取
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            // 记录特征点来源的图像金字塔图层
            keypoints[i].octave=level;
            // 记录计算方向的patch，缩放后对应的大小，又被称作为特征点半径
            keypoints[i].size = scaledPatchSize;
        }
    }

    // 计算每一层每一个特征点的主方向
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);

}



/*
 *  @brief      四叉树，均匀的选取特征点
 *  @param[in] vToDistributeKeys    当前层总的角点数量
 *  @param[in] minX    图像边界
 *  @param[in] maxX   
 *  @param[in] minY    
 *  @param[in] maxY    
 *  @param[in] N           该层所要提取特征点数
 *  @param[in] level    金字塔第level层数
*/
std::vector<cv::KeyPoint>  ORBextractor::DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // 1.Compute how many initial nodes   一般是1或2
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));  // 图像长/宽

    const float hX = static_cast<float>(maxX-minX)/nIni;    // 每个节点的列范围

    std::list<ExtractorNode> lNodes; // 节点list列表，类型ExtractorNode，存放节点

    std::vector<ExtractorNode*> vpIniNodes;  // 节点指针
    vpIniNodes.resize(nIni);    // 最初容量至少=初始节点nIni

    // 2.生成初始提取器节点
    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        // cv::Point2i 类的定义位于 <opencv2/core/types.hpp> 头文件中
        // 用于表示二维平面上的整数坐标点 i指int型  类模板Point_<T>
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0); 
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        // 把ni添加到 lNodes 容器的末尾
        lNodes.push_back(ni);
        // back() 是 list 容器的成员函数，用于获取容器中的最后一个元素
        // vpIniNodes节点指针 vpIniNodes[i]指针数组，指向当前节点
        // 存储这个初始的提取器节点句柄
        vpIniNodes[i] = &lNodes.back();
    }

    // 3. 把特征点分配给子节点(初始节点)
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        // 引用 kp
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        // kp.pt.x/hX初始化一般为1、2  存储到相应节点类成员对象vKeys
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    // 4 遍历此提取器节点列表，标记那些不可再分裂的节点，删除那些没有分配到特征点的节点
    std::list<ExtractorNode>::iterator lit = lNodes.begin();

    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)
        {
            lit->bNoMore=true; //标志位，表示此节点不可再分
            lit++;
        }
        else if(lit->vKeys.empty())
            // 如果一个节点没有被分配到特征点，那么就从列表中直接删除它
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    // 这个变量记录了在一次分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数目和其句柄(节点指针)
    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    // 调整大小，一个初始化节点将“分裂”成为四个节点
    vSizeAndPointerToNode.reserve(lNodes.size()*4);


    // 5 利用四叉树方法对图像进行划分区域，均匀分配特征点
    while(!bFinish)
    {
        iteration++;

        // 节点数
        int prevSize = lNodes.size();

        lit = lNodes.begin();
        // 有多少个节点展开，这个一直保持累计，不清零
        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)
            {
                // 5.1 当前节点只有一个特征点，不再分裂该特征点
                lit++;
                continue;
            }
            else
            {
                // 5.2 不止一个特征点, 1分为4
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                // 5.3 如果分裂的子节点区域特征点数比0大，接下面操作；否则该区域将被抛弃
                if(n1.vKeys.size()>0)
                {
                    // 只要这个节点区域内特征点数>=1,添加n1子节点
                    lNodes.push_front(n1); // 把这个节点放在容器第一个位置  执行循环否？
                    if(n1.vKeys.size()>1)
                    {
                        // 5.4 特征点数>1,说明还需要分裂(最终的目的就是每一个节点区域的特征点为1)
                        nToExpand++;
                        // 键值对数组容器  <n1节点特征点数，n1节点指针>
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                         // lNodes.front().lit 和前面的迭代的lit 不同，只是名字相同而已
                        // lNodes.front().lit是node结构体里的一个指针用来记录节点的位置
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit); // 删除分裂的母节点
                continue;
            }
        }       

        // 5.5 Finish 如果节点数>特征点数 or 所有的节点只有一个特征点
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true; // 结束循环
        }
        // 5.6 (int)lNodes.size()+nToExpand*3  实际分裂会得到节点数>N,那么选择特征点数比较大的节点分裂
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {

            while(!bFinish)
            {
                // 获取当前的list中的节点个数
                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                // 对需要划分的节点进行排序，对pair对的第一个元素(特征点数)进行排序，默认是从小到大排序
                // 优先分裂特征点多的节点，使得特征点密集的区域保留更少的特征点
                //! 注意这里的排序规则非常重要！会导致每次最后产生的特征点都不一样。建议使用 stable_sort
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());


                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    // 5.6.1 优先分裂特征点数比较大的节点
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // 5.6.2 Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    // 只要分裂的节点数>=N，那么直接跳出for循环，不在分裂节点
                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    // 7 到此，已经固定了最终的节点，但每个节点不一定只有一个特征点，我们保留每个节点区域响应值最大的一个特征点
    vector<cv::KeyPoint> vResultKeys;
    // 调整容器大小为要提取的特征点数目
    vResultKeys.reserve(nfeatures);
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        // 当前节点的特征点数组容器 引用
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        // 7.1 得到指向第一个特征点的指针，后面作为最大响应值对应的关键点
        cv::KeyPoint* pKP = &vNodeKeys[0];
        // 7.2 用第1个关键点响应值初始化最大响应值
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                // 7.3 只要这个关键点的相应大，记录其地址指针
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }
        // 7.4 记录每一个节点中响应最大的关键点
        vResultKeys.push_back(*pKP);

    //     // 画出图像中所有的节点
    //    if(level ==0)
    //    {
    //             cv::rectangle(mvImagePyramid[level],  lit->UL,  lit->BR, cv::Scalar(0, 255, 0), 2); // 绘制矩形
    //    }


    }
    // 8.返回最终关键点数组容器
    return vResultKeys;
}




/**
 * @brief 将提取器节点分成4个子节点，同时也完成图像区域的划分、特征点归属的划分，以及相关标志位的置位
 * 
 * @param[in & out] n1  提取器节点1：左上
 * @param[in & out] n2  提取器节点1：右上
 * @param[in & out] n3  提取器节点1：左下
 * @param[in & out] n4  提取器节点1：右下
 */
void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    // 1.计算当前节点所在图像区域的一半长宽  ceil向上取整  static_cast强制把Point2i类型转float   
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    // 2.计算子节点图像区域边界  1分为4  左上
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());	// reserve将容器的容量设置为至少vKeys.size
	
    // 右上 
    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());
	
    // 左下 
    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());
	
    // 右下
    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    // 把特征点分配到对应的4个节点区域
    for(size_t i=0;i<vKeys.size();i++)
    {	
        // 地址引用					
        const cv::KeyPoint &kp = vKeys[i];  // 注意vector数组容器中元素是cv::KeyPoint
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);		// 左上
            else
                n3.vKeys.push_back(kp);		// 右上
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);			// 左下
        else
            n4.vKeys.push_back(kp);			// 右下
    }

    // 判断这几个子节点所属特征点数量 == 1，若为1，表明该节点不再分裂
    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;
}



/*
 *  @brief      计算一个图像的描述子矩阵
 *  @param[in] image    特征点
 *  @param[in & out] keypoints    记录输入图像特征点
 *  @param[in & out] descriptors  记录输入图像特征点对应的描述子矩阵
 *  @param[in] pattern  计算描述子的随机点对--512对点
*/
void ORBextractor::computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                               const std::vector<cv::Point>& pattern)
{
    descriptors = cv::Mat::zeros((int)keypoints.size(), 32, CV_8UC1); // 行维=关键点总数  列维=32 矩阵元素都是CV_8UC1 8位  32*8=256

    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}


/*
 *  @brief      计算一个特征点的描述子向量
 *  @param[in] kpt    特征点
 *  @param[in] img    图像
 *  @param[in] pattern  计算描述子的随机点对--512对点
*/
void ORBextractor::computeOrbDescriptor(const cv::KeyPoint& kpt,
                                 const cv::Mat& img, const cv::Point* pattern,
                                 uchar* desc)
{
    float angle = (float)kpt.angle*factorPI;    // 角度转弧度    每个关键点主方向
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]


    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

    #undef GET_VALUE
}



/*
 *  @brief      特征提取总操作函数
 *  @param[in] image    原始图像    cv::InputArray 是 OpenCV 中用于传递输入数据的类。它可以接受多种类型的数据
 *  @param[in & out] _keypoints    记录了图像所有的特征点
 *  @param[in & out] _descriptors  记录输入图像特征点对应的描述子矩阵(rows = _keypoints.size   cols= 32 )  元素uchar
*/
void ORBextractor::operator()( cv::InputArray _image, std::vector<cv::KeyPoint>& _keypoints,
                      cv::OutputArray _descriptors)
{ 
    // 1 判断图像是否正确
    if(_image.empty())
        return;

    cv::Mat image = _image.getMat();
    assert(image.type() == CV_8UC1 );

    // 2 计算图像金字塔，金字塔保存到mvImagePyramid[level]
    ComputePyramid(image);

    // 3 计算图像的特征点，并且将特征点进行均匀化。均匀的特征点可以提高位姿计算精度
    // 存储所有的特征点，此处为二维的vector，第一维存储的是金字塔的层数，第二维存储的是那一层金字塔图像里提取的所有特征点
    vector < vector<KeyPoint> > allKeypoints;
    ComputeKeyPointsOctTree(allKeypoints);

    int nkeypoints = 0;
    // 计算所有的特征点总数nkeypoints（所有层）
    for (int level = 0; level < nlevels; ++level)
    {
        nkeypoints += (int)allKeypoints[level].size();
    }


    // 4 图像描述子矩阵descriptors
    cv::Mat descriptors;

    if( nkeypoints == 0 ) // 没有找到特征点
        // 调用cv::mat类的.realse方法，强制清空矩阵的引用计数，这样就可以强制释放矩阵的数据了
        _descriptors.release();
    else
    {
        // 如果图像金字塔中有特征点，那么就创建这个存储描述子的矩阵，注意这个矩阵是存储整个图像金字塔中特征点的描述子
        _descriptors.create(nkeypoints, 32, CV_8U); // nkeypoints是行数，32是列数
        descriptors = _descriptors.getMat();        // CV_8U矩阵元素的格式  32*8=256
    }                                               // getMat()获取这个描述子的矩阵信息

    _keypoints.clear();
    _keypoints.reserve(nkeypoints); // 预分配正确大小的空间

    // 因为遍历是一层一层进行的，但是描述子那个矩阵是存储整个图像金字塔中特征点的描述子，
    // 所以在这里设置了Offset变量来保存“寻址”时的偏移量，
    // 辅助进行在总描述子mat中的定位
    int offset = 0;
    for (int level = 0; level < nlevels; ++level)
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        // 当前金字塔层的特征点数
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel==0)
            continue;

        // preprocess the resized image
         // 5 对图像进行高斯模糊
        // 深拷贝当前金字塔所在层级的图像
        cv::Mat workingMat = mvImagePyramid[level].clone();
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // 计算当前层描述子
        // rowRange 函数，用于提取 descriptors 矩阵中指定行的子矩阵
        cv::Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        // workingMat高斯模糊之后的图层图像  keypoints当前图层中的特征点集合
        // desc当前层特征点对应的描述子矩阵  pattern随机点对模板   desc维度 = 当前层特征点数*32，每一个元素都是uchar 8位
        computeDescriptors(workingMat, keypoints, desc, pattern);

        offset += nkeypointsLevel;


        // 6 对非第0层图像中的特征点的坐标恢复到第0层图像（原图像）的坐标系下
        //  得到所有层特征点在第0层里的坐标放到_keypoints里面
        // 对于第0层的图像特征点，他们的坐标就不需要再进行恢复了
        if (level != 0)
        {
            float scale = mvscaleFactor[level]; // getScale(level, firstLevel, scaleFactor);
            for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        // 将keypoints中内容插入到_keypoints 的末尾
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}