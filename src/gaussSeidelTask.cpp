#include "gaussSeidelTask.h"
#include "gaussSeidel.h"
#include <omp.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <vector>
#include <unistd.h>
using namespace std;
// partie gauche de la diagonal
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
// partie droite de la diagonal
bool Task_Diag_Bot(const Mat m_Src_border, Mat &m_Dst_border,int rows,int cols){
    for(int row = 1; row < rows; row++){
        for(int col = cols - row - 50; col < cols; col++){
            for(int chanel = 0; chanel < 3; chanel++){
                m_Dst_border.at<Vec3b>((row + 1),(col + 1))[chanel] = (m_Dst_border.at<Vec3b>((row + 1) - 1,(col + 1))[chanel] +
                m_Dst_border.at<Vec3b>((row + 1),(col + 1) - 1)[chanel] + m_Src_border.at<Vec3b>((row + 1) + 1,(col + 1))[chanel] +
                m_Src_border.at<Vec3b>((row + 1),(col + 1) + 1)[chanel] + m_Src_border.at<Vec3b>((row + 1),(col + 1))[chanel]) / 5;
            }
        }
    }
    return true;
}
// Partie parallél
bool AddGaussSeidelDiag(const Mat m_Src, Mat &m_Dst, int it){

    // Variables Declaration
    int rows = m_Src.rows, cols = m_Src.cols;
    Mat m_Tmp_border(m_Src.rows + 1,m_Src.cols + 1,m_Src.type());

    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    
    // timer
    double start_t, end_t;

    vector<Mat> tmp_vec(it+1);
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

    //
    tmp_vec[0] = m_Tmp_border;
    
    for (int i = 0; i < it; i++) {
        Mat img = m_Tmp_border.clone();
        dst_vec[i] = img;
    }
    Mat tmp;
    Mat* next_top_ptr;
    Mat* prev_top_ptr;
    Mat* next_bot_ptr;
    Mat* prev_bot_ptr;
    start_t = omp_get_wtime();
    // Parallélisation
    #pragma omp parallel
    #pragma omp single
    {
        
        
        for(int i = 0; i < it; i++){
            // denoiser (parallel part)
            int prev;
            if(i == 0) prev = 0;
            else prev = i - 1;
            prev_top_ptr = &dst_vec[prev];
            next_top_ptr = &dst_vec[i];
            prev_bot_ptr = &tmp_vec[i];
            next_bot_ptr = &tmp_vec[i+1];
            if(i == 0){
                #pragma omp task depend(out: next_top_ptr)
                {
                    Task_Diag_Top(tmp_vec[i],dst_vec[i],rows,cols);
                }
                #pragma omp task depend(in: next_top_ptr) depend(out: next_bot_ptr)
                {
                    
                    tmp_vec[i+1] = dst_vec[i];
                    Task_Diag_Bot(tmp_vec[i],tmp_vec[i+1],rows,cols);
                }
            }
            else{
                #pragma omp task depend(in: prev_top_ptr) depend(out: next_top_ptr)
                    {
                        Task_Diag_Top(dst_vec[i-1],dst_vec[i],rows,cols);
                        
                    }
                    #pragma omp task depend(in: prev_bot_ptr) depend(out: next_bot_ptr)
                    {
                        tmp_vec[i+1] = dst_vec[i];
                        Task_Diag_Bot(tmp_vec[i],tmp_vec[i+1],rows,cols);

                    }
            }
        }
        #pragma omp taskwait
    }
    end_t = omp_get_wtime();
    cout << string(100, '-') << endl;
    cout << "| Parallel version of Diagonal approch version took " << end_t - start_t << " seconds." << endl;
    cout << string(100, '-') << endl;
    // copie de l'image dans la matrice de destination
    copyFromBorder(tmp_vec[it], m_Dst);
    m_Tmp_border.release();
    return true;
}
