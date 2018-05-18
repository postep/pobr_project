#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <math.h>

void hsv_desaturation(cv::Mat &I){
    cv::Mat_<cv::Vec3b> _I = I;
    for(int r = 0; r < I.rows; ++r){
        for(int c = 0; c < I.cols; ++c){
            _I(r, c)[1] = 0;
        }
    }

    I = _I;
}



int recognition(cv::Mat I, cv::Mat &R){
    R = I.clone();
    cv::Mat_<cv::Vec3b> _I = I;
    cv::Mat HSV;
    cvtColor(I, HSV, CV_BGR2HSV);
    hsv_desaturation(HSV);
    cvtColor(HSV, R, CV_HSV2BGR);
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