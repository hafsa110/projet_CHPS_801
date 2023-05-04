#include "gaussSeidel.h"
#include <omp.h>
#include <cstdlib>
#include <algorithm>
using namespace std;

bool AddGaussSeidel(const Mat m_Src, Mat &m_Dst)
{
    // Variables Declaration
    Mat m_Src_border(m_Src.rows,m_Src.cols,m_Src.type());
    int border_type = BORDER_CONSTANT;
    int size_border = 0;

    // Verify if the source image is empty
    if(m_Src.empty())
    {
        cout<<"[Error]! Input Image Empty!";
        return 0;
    }
    
    // Add a border of zeros to the source matrix
    copyMakeBorder(m_Src,m_Src_border,size_border,size_border,size_border,size_border,border_type,0);

    // itterat over each pixel of the image
    for(int row = 1; row < m_Src.rows - 1; ++row){

        for(int col = 1; col < m_Src.cols - 1; ++col){

            // calculate the value of a pixel for each channel following the gaussSeidel method
            for(int chanel = 0; chanel < 3; chanel++){
                
                m_Dst.at<Vec3b>(row,col)[chanel] = (m_Dst.at<Vec3b>(row - 1,col)[chanel] +
                m_Dst.at<Vec3b>(row,col - 1)[chanel] + m_Src_border.at<Vec3b>(row + 1,col)[chanel] +
                m_Src_border.at<Vec3b>(row,col + 1)[chanel] + m_Src_border.at<Vec3b>(row,col)[chanel]) / 5;
            }
        }
    }
    
    return true;
}

bool AddCopy(const Mat m_Src, Mat &m_Dst){
    int rows = m_Src.rows;
    int cols = m_Src.cols;
    int row,col;
    int nb_elements;
    int i_1,i_2,j_1,j_2;
    // 
    int number_diag = rows + cols - 1;
    
    for(int current_diag = 0; current_diag < number_diag ;current_diag++){
        if(current_diag < rows){
            i_1 = current_diag;
            j_1 = 0;
        }
        else{
            j_1 = current_diag - (rows - 1);
            i_1 = rows - 1;
        }
        if(current_diag < cols){
            j_2 = current_diag;
            i_2 = 0;
        }
        else{
            i_2 = current_diag - (cols - 1);
            j_2 = cols - 1;
        }
        nb_elements = min(abs(i_1 - i_2),abs(j_1 - j_2));
        for(int k = 0; k < nb_elements; k++){
            row = i_1 - k;
            col = j_1 + k;
            for(int chanel = 0; chanel < 3; chanel++){
                m_Dst.at<Vec3b>(row,col)[chanel] = m_Src.at<Vec3b>(row,col)[chanel];
            }
        }
    }
    
    return true;
}
// Applies gauss blur using tasks
bool AddGaussSeidel_parallel(const Mat m_Src, Mat &m_Dst){
    return true;
}