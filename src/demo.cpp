#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <math.h>

int recognition(cv::Mat I, cv::Mat &R){
    R = I.clone();
    cv::Mat_<cv::Vec3b> _I = I;
    return 0;
}

void perform(std::string name){
    cv::Mat I = cv::imread(name);
    CV_Assert(I.depth() != sizeof(uchar));
    if(I.channels() == 3){
        std::cout << "RECOGNITION: " << name << std::endl;
        cv::Mat R;
        recognition(I, R);
        cv::imshow("loaded " + name, I);
        cv::imshow("recognized" + name, R);
    }else{
        std::cout << "WRONG IMAGE" << std::endl;
    }
}


int main(int, char *[]) {
    perform("./images/caparol.jpg");
    cv::waitKey(-1);
    return 0;
}