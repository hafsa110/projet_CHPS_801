#include "jacobi.h"

bool AddJacobi(const Mat m_Src, Mat &m_Dst)
{
    Mat m_Src_border(m_Src.rows,m_Src.cols,m_Src.type());
    int border_type = BORDER_CONSTANT;
    int size_border = 1;


    // Verify if the source image is empty
    if(m_Src.empty())
    {
        cout<<"[Error]! Input Image Empty!";
        return 0;
    }

    copyMakeBorder(m_Src,m_Src_border,size_border,size_border,size_border,size_border,border_type,0);

    
    // itterat over each pixel of the image
    for(int row = 1; row <= m_Src.rows; ++row){
        for(int col = 1; col <= m_Src.cols; ++col){
            // calculate the value of a pixel for each channel following the gaussSeidel method
            for(int chanel = 0; chanel < 3; chanel++){

                m_Dst.at<Vec3b>(row - 1,col - 1)[chanel] = (m_Src_border.at<Vec3b>(row - 1,col)[chanel] +
                m_Src_border.at<Vec3b>(row,col - 1)[chanel] + m_Src_border.at<Vec3b>(row + 1,col)[chanel] +
                m_Src_border.at<Vec3b>(row,col + 1)[chanel] + m_Src_border.at<Vec3b>(row,col)[chanel]) / 5;
            }
        }
    }
    
    
    return true;
}