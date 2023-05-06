#include "gaussSeidelTask.h"
#include "gaussSeidel.h"
#include <omp.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <vector>
using namespace std;

bool Task_Diag_Top(const Mat m_Src_border, Mat &m_Dst_border,int rows,int cols){
    for(int row = 0; row < rows; row++){
        
        for(int col = 0; col < cols - row; col++){
            for(int chanel = 0; chanel < 3; chanel++){
                m_Dst_border.at<Vec3b>((row + 1),(col + 1))[chanel] = (m_Dst_border.at<Vec3b>((row + 1) - 1,(col + 1))[chanel] +
                m_Dst_border.at<Vec3b>((row + 1),(col + 1) - 1)[chanel] + m_Src_border.at<Vec3b>((row + 1) + 1,(col + 1))[chanel] +
                m_Src_border.at<Vec3b>((row + 1),(col + 1) + 1)[chanel] + m_Src_border.at<Vec3b>((row + 1),(col + 1))[chanel]) / 5;
            }
        }
    }
    return true;
}
bool Task_Diag_Bot(const Mat m_Src_border, Mat &m_Dst_border,int rows,int cols){
    for(int row = 1; row < rows; row++){
        for(int col = cols - row; col < cols; col++){
            for(int chanel = 0; chanel < 3; chanel++){
                m_Dst_border.at<Vec3b>((row + 1),(col + 1))[chanel] = (m_Dst_border.at<Vec3b>((row + 1) - 1,(col + 1))[chanel] +
                m_Dst_border.at<Vec3b>((row + 1),(col + 1) - 1)[chanel] + m_Src_border.at<Vec3b>((row + 1) + 1,(col + 1))[chanel] +
                m_Src_border.at<Vec3b>((row + 1),(col + 1) + 1)[chanel] + m_Src_border.at<Vec3b>((row + 1),(col + 1))[chanel]) / 5;
            }
        }
    }
    return true;
}

bool AddGaussSeidelDiag(const Mat m_Src, Mat &m_Dst, int it){

    // Variables Declaration
    int rows = m_Src.rows, cols = m_Src.cols;
    Mat m_Tmp_border(m_Src.rows + 1,m_Src.cols + 1,m_Src.type());
    Mat m_Dst_border(m_Src.rows + 1,m_Src.cols + 1,m_Src.type());
    // timer
    double start_t, end_t;

    vector<Mat> tmp_vec(it);
    vector<Mat> dst_vec(it);
    int border_type = BORDER_CONSTANT;
    int size_border = 1;

    // Verify if the source image is empty
    if(m_Src.empty())
    {
        cout<<"[Error]! Input Image Empty!";
        return 0;
    }
    // Add a border of zeros to the source matrix
    copyMakeBorder(m_Src,m_Tmp_border,size_border,size_border,size_border,size_border,border_type,0);
    copyMakeBorder(m_Dst,m_Dst_border,size_border,size_border,size_border,size_border,border_type,0);
    //
    tmp_vec[0] = m_Tmp_border;
    dst_vec[0] = m_Dst_border;
    Mat* tmp_ptr;
    Mat* dst_ptr;
    start_t = omp_get_wtime();
    #pragma omp parallel
    #pragma omp single
    {
        #pragma omp taskgroup
        {
            for(int i = 0; i < it; i++){
                // denoiser (parallel part)
                tmp_ptr = &tmp_vec[i];
                dst_ptr = &dst_vec[i];
                #pragma omp task depend(inout: tmp_ptr) depend(inout: dst_ptr)
                {
                    Diag_Top(tmp_vec[i],dst_vec[i],rows,cols);
                    Diag_Bot(tmp_vec[i],dst_vec[i],rows,cols);
                }
                #pragma omp taskwait
                if(i != it - 1){
                    tmp_vec[i+1] = dst_vec[i];
                    dst_vec[i+1] = dst_vec[i];
                }
                
            }
        }
        
    }
    end_t = omp_get_wtime();
    cout << string(100, '-') << endl;
    cout << "| Parallel version of Diagonal approch version took " << end_t - start_t << " seconds." << endl;
    cout << string(100, '-') << endl;
    // copie de l'image dans la matrice de destination
    copyFromBorder(dst_vec[it - 1], m_Dst);
    m_Dst_border.release();
    m_Tmp_border.release();
    return true;
}