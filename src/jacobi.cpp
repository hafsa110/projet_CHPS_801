#include "Jacobi.h"

bool AddJacobi(const Mat m_Src, Mat &m_Dst)
{
    // Verify if the source image is empty
    if(m_Src.empty())
    {
        cout<<"[Error]! Input Image Empty!";
        return 0;
    }
    
    // itterat over each pixel of the image
    for(int row = 1; row < m_Src.rows - 1; ++row){
        for(int col = 1; col < m_Src.cols - 1; ++col){
            // calculate the value of a pixel for each channel following the gaussSeidel method
            for(int chanel = 0; chanel < 3; chanel++){
                
                m_Dst.at<Vec3b>(row,col)[chanel] = (m_Src.at<Vec3b>(row - 1,col)[chanel] +
                m_Src.at<Vec3b>(row,col - 1)[chanel] + m_Src.at<Vec3b>(row + 1,col)[chanel] +
                m_Src.at<Vec3b>(row,col + 1)[chanel] + m_Src.at<Vec3b>(row,col)[chanel]) / 5;
            }
        }
    }
    
    return true;
}