#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <math.h>

int min(int a, int b){
    return (a < b) ? a : b;
}

int max(int a, int b){
    return (a < b) ? b : a;
}

int normalize(double v){
    if (v > 255){
        return 255;
    }else if(v < 0){
        return 0;
    }else{
        return v;
    }
}

void hsv_desaturation(cv::Mat &I){
    cv::Mat_<cv::Vec3b> _I = I;
    for(int r = 0; r < I.rows; ++r){
        for(int c = 0; c < I.cols; ++c){
            _I(r, c)[1] = 0;
        }
    }

    I = _I;
}

void convolution(cv::Mat &I, double (*f)[5]){
    cv::Mat_<cv::Vec3b> _I = I;
    cv::Mat_<cv::Vec3b> _R = I.clone();
    int size = 5;
    double ratio = 0;
    for(int i = 0; i < size; ++i){
        for(int j = 0; j < size; ++j){
            ratio += f[i][j];
        }
    }
    
    for(int r = 0; r < I.rows; ++r){
        for(int c = 0; c < I.cols; ++c){
            
            double cr = 0, cg = 0, cb = 0;
            for(int p = 0; p <  size; ++p){
                for(int q = 0; q < size; ++q){
                    if(r+p-(size/2) >= 0 && r+p-(size/2) < I.rows && 
                                c+q-(size/2) >= 0 && c+q-(size/2) < I.cols){
                        cb += f[p][q]*_I(r+p-(size/2), c+q-(size/2))[0];
                        cg += f[p][q]*_I(r+p-(size/2), c+q-(size/2))[1];
                        cr += f[p][q]*_I(r+p-(size/2), c+q-(size/2))[2];    
                    }
                }
            }
            cb /= ratio;
            cg /= ratio;
            cr /= ratio;

            _R(r, c)[0] = normalize((int)cb);
            _R(r, c)[1] = normalize((int)cg);
            _R(r, c)[2] = normalize((int)cr);
        }
    }

    I = _R;
}


double func(double f[5][5][3], int idx){
    double res = f[0][0][idx];
    for(int i = 0; i < 5; ++i){
        for(int j = 0; j < 5; ++j){
            if(res < f[i][j][idx]){
                res = f[i][j][idx];
            }
        }
    }
    return res;
}

void binary_filter(cv::Mat &I){
    int size = 5;
    cv::Mat_<cv::Vec3b> _I = I;
    cv::Mat_<cv::Vec3b> _R = I.clone();
    for(int r = 0; r < I.rows; ++r){
        for(int c = 0; c < I.cols; ++c){
            
            double f[5][5][3];
            for(int p = 0; p <  size; ++p){
                for(int q = 0; q < size; ++q){
                    if(r+p-(size/2) >= 0 && r+p-(size/2) < I.rows && 
                                c+q-(size/2) >= 0 && c+q-(size/2) < I.cols){
                        f[p][q][0] = _I(r+p-(size/2), c+q-(size/2))[0];
                        f[p][q][1] = _I(r+p-(size/2), c+q-(size/2))[1];
                        f[p][q][2] = _I(r+p-(size/2), c+q-(size/2))[2];    
                    }else{
                        f[p][q][0] = 0;
                        f[p][q][1] = 0;
                        f[p][q][2] = 0;
                    }
                }
            }

            _R(r, c)[0] = normalize((int)func(f, 0));
            _R(r, c)[1] = normalize((int)func(f, 1));
            _R(r, c)[2] = normalize((int)func(f, 2));
        }
    }

    I = _R;
}

void gaussian_blur(cv::Mat &I){
    double matrix[5][5] = {1, 4, 7, 4, 1,
        4, 16, 16, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1, 4, 7, 4, 1};

    convolution(I, matrix);
}
int recognition(cv::Mat I, cv::Mat &R){
    R = I.clone();
    cv::Mat_<cv::Vec3b> _I = I;
    cv::Mat HSV;
    cvtColor(I, HSV, CV_BGR2HSV);
    hsv_desaturation(HSV);
    gaussian_blur(HSV);
    gaussian_blur(HSV);
    
    binary_filter(HSV);
    
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