#include <iostream>
#include <vector>
#include <cstdlib>      // rand()
#include <ctime>
#include <cmath>
#include <random>

using namespace std;





struct Point{
    Point(double x, double y):_x(x), _y(y){
        // cout << endl;
    }
    ~Point(){}
    double _x;
    double _y;
};


struct Line{
    Line(double A, double B, double C):_A(A), _B(B), _C(C){
        // cout << endl;
    }
    ~Line(){}

    double _A, _B, _C;

    double compute_distance_squared(double x, double y){
        double distance_squared = pow(_A * x + _B * y + _C, 2) / (_A * _A + _B * _B);
        return distance_squared;
    }
};


struct Line2{
    Line2(double k, double b):_k(k), _b(b){
        // cout << endl;
    }
    ~Line2(){}

    double _k, _b;

    double compute_distance_squared(double x, double y){
        double distance_squared = pow(y - _k * x - _b, 2) ;
        return distance_squared;
    }
};

vector<double> RANSAC_Line(vector<Point> &points, int iter, double threshold){
    int n = points.size();
    if (n < 2) {
        cerr << "Not Enough Points" <<endl;
        return vector<double>();
    }

    int best_points = 0;
    vector<double> best(4);

    for(int i = 0; i < iter; i++){
        
        // 1. 随机选择两个点
        int id1 = rand() % n;   // 取余，生成一个区间位于0-n的数
        int id2 = rand() % n;
        while (id1 == id2) {
            id2 = rand() % n;
        }

        Point p1 = points[id1];
        Point p2 = points[id2];

        // 2. 拟合直线模型---Ax+By+C=0
        double A = p1._y - p2._y;       // A = Y1 - Y2
        double B = p2._x - p1._x;       // B = X2 - X1
        double C = p1._x * p2._y - p2._x * p1._y;       // C = X1*Y2 - X2*Y1

        Line L(A, B, C);

        // 3. 计算距离，统计内点数量
        int count_inlier = 0;
        for(int j = 0; j < n; j++){
            if(j != id1 && j != id2){
                double distance_squared = L.compute_distance_squared(points[j]._x, points[j]._y);
                if(distance_squared < threshold){
                    count_inlier++;
                }
            }
        }
        cout << "当前是第 " << i << "迭代，内点总数是：" << count_inlier << endl; 
        // toDo:增加一个内点集合阈值，超过这个阈值，即认为模型满足条件，跳出

        // 4. 记录最好的情况 
        if(count_inlier > best_points){
            best_points = count_inlier;
            best[0] = A;
            best[1] = B;
            best[2] = C;
            best[3] = best_points;
        }
    }
    return best;
}

vector<double> RANSAC_Line2(vector<Point> &points, int iter, double threshold){
    int n = points.size();
    if (n < 2) {
        cerr << "Not Enough Points" <<endl;
        return vector<double>();
    }

    int best_points = 0;
    vector<double> best(3);

    for(int i = 0; i < iter; i++){
        
        // 1. 随机选择两个点
        int id1 = rand() % n;   // 取余，生成一个区间位于0-n的数
        int id2 = rand() % n;
        while (id1 == id2) {
            id2 = rand() % n;
        }

        Point p1 = points[id1];
        Point p2 = points[id2];

        // 2. 拟合直线模型---Ax+By+C=0
        double A = p2._y - p1._y;       //
        double B = p2._x - p1._x;       // 
        double k = A / B;       // 
        double b = p2._y - k * p2._x;

        Line2 L(k, b);

        // 3. 计算距离，统计内点数量
        int count_inlier = 0;
        for(int j = 0; j < n; j++){
            if(j != id1 && j != id2){
                double distance_squared = L.compute_distance_squared(points[j]._x, points[j]._y);
                if(distance_squared < threshold){
                    count_inlier++;
                }
            }
        }
        // cout << "当前是第 " << i << "迭代，内点总数是：" << count_inlier << endl; 

	// toDo:增加一个内点集合阈值，超过这个阈值，即认为模型满足条件，跳出


        // 4. 记录最好的情况 
        if(count_inlier > best_points){
            best_points = count_inlier;
            best[0] = k;
            best[1] = b;
            best[2] = best_points;
        }
    }
    return best;
}


std::vector<Point> generateRandomPoints(int numPoints, double A, double B, double C, double mean,double stddev) {
    std::vector<Point> points;
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<double> distribution(mean, stddev);
    std::srand(std::time(nullptr));
    for (int i = 0; i < numPoints; ++i) {
        // 生成随机的 x 坐标
        double x = static_cast<double>(rand()) / RAND_MAX * 10; // 假设 x 的范围在 [0, 10]

        // 根据直线方程计算对应的 y 坐标
        double y = (-A * x - C) / B;

        // 添加随机扰动以模拟测量误差
        y += distribution(generator);
        cout << x << ' ' << y << endl;
        points.push_back(Point(x, y));
    }
    return points;
}

std::vector<Point> generateRandomPoints2(int numPoints, double k, double b, double mean,double stddev) {
    std::vector<Point> points;
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<double> distribution(mean, stddev);
    std::srand(std::time(nullptr));
    for (int i = 0; i < numPoints; ++i) {
        // 生成随机的 x 坐标
        double x = static_cast<double>(rand()) / RAND_MAX * 10; // 假设 x 的范围在 [0, 10]

        // 根据直线方程计算对应的 y 坐标
        double y = k * x + b;

        // 添加随机扰动以模拟测量误差
        y += distribution(generator);
        // cout << x << ' ' << y << endl;
        points.push_back(Point(x, y));
    }
    return points;
}


int main() {
    double A = -2, B = 1, C = 1;
    double k = 4, b = 1;
    // 生成随机点集
    // vector<Point> points = generateRandomPoints(50, -1, 1, 1, 0, 0.2); // 生成200个随机点
    vector<Point> points = generateRandomPoints2(50, k, b, 0, 0.2);

    // 调用 RANSAC 算法拟合直线
    // double threshold = 3.84 * 0.02 * 0.02;
    double threshold = 0.05;
    // vector<double> ABC_SIM = RANSAC_Line(points, 1000, threshold);
    vector<double> ABC_SIM = RANSAC_Line2(points, 1000, threshold);
    cout << "k :"<<ABC_SIM[0]  << " ,b:" << ABC_SIM[1] << " 内点数:" << ABC_SIM[2] ;
    
    // for(auto &param:ABC_SIM){
    //     cout << param << ' ';
    // }
    return 0;
}


