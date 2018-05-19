#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <vector>
#include <queue>
#include <math.h>

class Rect{
public:
    int min_row;
    int max_row;
    int min_col;
    int max_col;
    bool text;
    Rect(int r, int c){
        min_row = r;
        max_row = r;
        min_col = c;
        max_col = c;
        text = true;
    }
    void spread(int r, int c){
        if(min_row > r){
            min_row = r;
        }
        if(max_row < r){
            max_row = r;
        }
        if(min_col > c){
            min_col = c;
        }
        if(max_col < c){
            max_col = c;
        }
    }

    int get_lower_row(){
        return min_row + 2*(max_row-min_row)/3;
    }

    int get_higher_row(){
        return min_row + 1*(max_row-min_row)/3;   
    }

    int get_lower_col(){
        return min_col + 2*(max_col-min_col)/3;
    }

    int get_higher_col(){
        return min_col + 1*(max_col-min_col)/3;   
    }

    void print(){
        std::cout << "min_row: " << min_row << " max_row: " << max_row 
        << " min_col: " << min_col << " max_col:" << max_col;
    }
};


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

double mask_max(double f[5][5][3], int idx){
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


double mask_min(double f[5][5][3], int idx){
    double res = f[0][0][idx];
    for(int i = 0; i < 5; ++i){
        for(int j = 0; j < 5; ++j){
            if(res > f[i][j][idx]){
                res = f[i][j][idx];
            }
        }
    }
    return res;
}

void binary_filter(cv::Mat &I, int type){
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

            switch(type){
                case 0:
                {
                    _R(r, c)[0] = normalize((int)mask_min(f, 0));
                    _R(r, c)[1] = normalize((int)mask_min(f, 1));
                    _R(r, c)[2] = normalize((int)mask_min(f, 2));
                    break;
                }
                case 1:
                {
                    _R(r, c)[0] = normalize((int)mask_max(f, 0));
                    _R(r, c)[1] = normalize((int)mask_max(f, 1));
                    _R(r, c)[2] = normalize((int)mask_max(f, 2));
                    break;   
                }
            }
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

void erosion(cv::Mat &I){
    binary_filter(I, 0);
}

void dilation(cv::Mat &I){
    binary_filter(I, 1);
}


void treshold(cv::Mat &I, int t0, int t1, int t2){
    cv::Mat_<cv::Vec3b> _I = I;

    for(int r = 0; r < I.rows; ++r){
        for(int c = 0; c < I.cols; ++c){
            if(t0 >= 0 && t0 <=255){
                if(_I(r, c)[0] < t0){
                    _I(r, c)[0] = 0;
                }else{
                    _I(r, c)[0] = 255;
                }
            }
            if(t1 >= 0 && t1 <=255){
                if(_I(r, c)[1] <= t1){
                    _I(r, c)[1] = 0;
                }else{
                    _I(r, c)[1] = 255;
                }
            }
            if(t2 >= 0 && t2 <=255){
                if(_I(r, c)[2] < t2){
                    _I(r, c)[2] = 0;
                }else{
                    _I(r, c)[2] = 255;
                }    
            }
        }
    }

    I = _I;
}

void draw_rect(cv::Mat &I, int min_row, int min_col, int max_row, int max_col, int cb, int cg, int cr){
    cv::Mat_<cv::Vec3b> _I = I;

    for(int i = min_row; i <= max_row; ++i){
        _I(i, min_col)[0] = cb;
        _I(i, min_col)[1] = cg;
        _I(i, min_col)[2] = cr;
        _I(i, max_col)[0] = cb;
        _I(i, max_col)[1] = cg;
        _I(i, max_col)[2] = cr;
    }

    for(int i = min_col; i <= max_col; ++i){
        _I(min_row, i)[0] = cb;
        _I(min_row, i)[1] = cg;
        _I(min_row, i)[2] = cr;
        _I(max_row, i)[0] = cb;
        _I(max_row, i)[1] = cg;
        _I(max_row, i)[2] = cr;
    }
    
    I = _I;
}


Rect bfs(cv::Mat &I, cv::Mat_<cv::Vec3b> &_I, int r, int c){
    Rect rect(r, c);
    _I(r, c)[2] = 255;
    std::queue<std::pair<int, int>> q;
    q.push(std::pair<int, int>(r, c));
    while(!q.empty()){
        std::pair<int, int> p = q.front();
        q.pop();
        int rq = p.first;
        int cq = p.second;
        
        if(rq-1 >= 0 && _I(rq-1, cq)[2] == 0){
            rect.spread(rq-1, cq);
            _I(rq-1, cq)[2] = 255;
            q.push(std::pair<int, int>(rq-1, cq));    
        }
        
        if(cq-1 >= 0 && _I(rq, cq-1)[2] == 0){
            rect.spread(rq, cq-1);
            _I(rq, cq-1)[2] = 255;
            q.push(std::pair<int, int>(rq, cq-1));    
        }

        if(rq+1 < I.rows && _I(rq+1, cq)[2] == 0){
            rect.spread(rq+1, cq);
            _I(rq+1, cq)[2] = 255;
            q.push(std::pair<int, int>(rq+1, cq));    
        }

        if(cq+1 < I.cols && _I(rq, cq+1)[2] == 0){
            rect.spread(rq, cq+1);
            _I(rq, cq+1)[2] = 255;
            q.push(std::pair<int, int>(rq, cq+1));    
        }
    }
    return rect;
}


std::vector<Rect> detect_shapes(cv::Mat &I){
    std::vector<Rect> rects;
    cv::Mat_<cv::Vec3b> _I = I;
    for(int r = 0; r < I.rows; ++r){
        for(int c = 0; c < I.cols; ++c){
            if(_I(r, c)[2] == 0){
                rects.push_back(bfs(I, _I, r, c));
            }
        }
    }
    return rects;
}


bool angle_correct(int value, int nominal){
    int diff = abs(value-nominal); 
    if(diff > 90){
        diff =  abs(diff-180);
    }
    return diff < 5;
}

bool stripe(int v, int stripe_n){
    switch(stripe_n){
        case 0: return angle_correct(v, 161);
        case 1: return angle_correct(v, 100);
        case 2: return angle_correct(v, 60);
        case 3: return angle_correct(v, 20);
        case 4: return angle_correct(v, 1);
        case 5: return angle_correct(v, 180);
    }
}

bool check_row(cv::Mat_<cv::Vec3b> _I, int min_col, int max_col, int row){
    int state = 0;
    for(int c = min_col; c < max_col; ++c){
        state += stripe(_I(row, c)[0], state);
    }
    return state > 5;
}

bool check_col(cv::Mat_<cv::Vec3b> _I, int min_row, int max_row, int col){
    int state = 0;
    for(int r = min_row; r < max_row; ++r){
        state += stripe(_I(r, col)[0], state);
    }
    return state > 5;
}

bool check_rising(cv::Mat_<cv::Vec3b> _I, int min_row, int max_row, int col){
    int state = 0;
    for(int r = min_row; r < max_row; ++r){
        state += stripe(_I(r, (min_row-r) + col)[0], state);
    }
    return state > 5;
}

bool check_falling(cv::Mat_<cv::Vec3b> _I, int min_row, int max_row, int col){
    int state = 0;
    for(int r = min_row; r < max_row; ++r){
        state += stripe(_I(r, -(min_row-r) + col)[0], state);
    }
    return state > 5;
}


std::vector<Rect> detect_caparols(cv::Mat HSV, std::vector<Rect> bounds){
    std::vector<Rect> caparols;
    cv::Mat_<cv::Vec3b> _I = HSV;
    for(int i = 0; i < bounds.size(); ++i){
        Rect r = bounds[i];
        bool caparol = check_row(_I, r.min_col, r.max_col, r.get_higher_row());
        caparol = caparol || check_row(_I, r.min_col, r.max_col, r.get_lower_row());
        caparol = caparol || check_col(_I, r.min_row, r.max_row, r.get_higher_col());
        caparol = caparol || check_col(_I, r.min_row, r.max_row, r.get_lower_col());
        caparol = caparol || check_rising(_I, r.min_row, r.max_row, r.get_lower_col());
        caparol = caparol || check_rising(_I, r.min_row, r.max_row, r.get_higher_col());
        caparol = caparol || check_falling(_I, r.min_row, r.max_row, r.get_lower_col());
        caparol = caparol || check_falling(_I, r.min_row, r.max_row, r.get_higher_col());

        if(caparol && !(r.max_col == HSV.cols-1 && r.min_col == 0 && r.min_row == 0 && r.max_row == HSV.rows-1)){
            caparols.push_back(r);
        }
    }
    return caparols;
}

std::vector<Rect> recognition(cv::Mat &I){
    
    cv::Mat HSV;
    cvtColor(I, HSV, CV_BGR2HSV);
    cv::Mat HSV2 = HSV.clone();
    
    treshold(HSV, -1, 255, -1);
    gaussian_blur(HSV);
    erosion(HSV);
    gaussian_blur(HSV);
    erosion(HSV);
    erosion(HSV);
    erosion(HSV);
    erosion(HSV);
    dilation(HSV);
    dilation(HSV);
    dilation(HSV);
    gaussian_blur(HSV);
    treshold(HSV, -1, -1, 60);

    std::vector<Rect> bounds = detect_shapes(HSV);
    std::vector<Rect> caparols = detect_caparols(HSV2, bounds);

    return caparols;
}

void perform(std::vector<std::string> names){
    for(int i = 0; i < names.size(); ++i){
        cv::Mat I = cv::imread(names[i]);
        CV_Assert(I.depth() != sizeof(uchar));
        if(I.channels() == 3){
        
            std::cout << "RECOGNITION: " << names[i] << std::endl;
            std::vector<Rect> caparols = recognition(I);
            std::cout << "RECOGNIZED: " << caparols.size() << std::endl;

            cv::Mat R = I.clone();
            for(int i = 0; i < caparols.size(); i++){
                Rect r = caparols[i]; 
                r.print();
                std::cout << std::endl;
                if(r.text){
                    draw_rect(R, r.min_row, r.min_col, r.max_row, r.max_col, 255, 0, 0);
                }else{
                    draw_rect(R, r.min_row, r.min_col, r.max_row, r.max_col, 0, 0, 255);
                }
            }
            cv::imshow("PROCESSED: " + names[i], R);
        
        }else{
            std::cout << "WRONG IMAGE: " << names[i] << std::endl;
        }
    }
}

int main(int, char *[]) {
    std::vector<std::string> names;
    names.push_back("./images/caparol.jpg");
    perform(names);
    cv::waitKey(-1);
    return 0;
}