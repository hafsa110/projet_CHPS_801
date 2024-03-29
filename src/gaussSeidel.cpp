#include "gaussSeidel.h"
#include <omp.h>
#include <cstdlib>
#include <algorithm>
using namespace std;

bool AddGaussSeidel(const Mat m_Src, Mat &m_Dst, int it){
    // Variables Declaration
    int rows = m_Src.rows, cols = m_Src.cols;
    Mat m_Tmp_border(m_Src.rows + 1,m_Src.cols + 1,m_Src.type());
    Mat m_Dst_border(m_Src.rows + 1,m_Src.cols + 1,m_Src.type());
    // timer
    double start_t, end_t;
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
    start_t = omp_get_wtime();
    for(int i = 0; i < it; i++)
    {
        // itterat over each pixel of the image
        for(int row = 0; row < rows; row++){
            for(int col = 0; col < cols; col++){
                for(int chanel = 0; chanel < 3; chanel++){
                    m_Dst_border.at<Vec3b>((row + 1),(col + 1))[chanel] = (m_Dst_border.at<Vec3b>((row + 1) - 1,(col + 1))[chanel] +
                    m_Dst_border.at<Vec3b>((row + 1),(col + 1) - 1)[chanel] + m_Tmp_border.at<Vec3b>((row + 1) + 1,(col + 1))[chanel] +
                    m_Tmp_border.at<Vec3b>((row + 1),(col + 1) + 1)[chanel] + m_Tmp_border.at<Vec3b>((row + 1),(col + 1))[chanel]) / 5;
                }
            }
        }
        m_Tmp_border = m_Dst_border;
    }
    end_t = omp_get_wtime();
    cout << string(100, '-') << endl;
    cout << "| Gauss Seidel naif version took " << end_t - start_t << " seconds." << endl;
    cout << string(100, '-') << endl;
    copyFromBorder(m_Dst_border, m_Dst);
    return true;
}
//
bool AddGaussSeidel_wave(const Mat m_Src, Mat &m_Dst){
    // Déclaration des variables
    int rows = m_Src.rows, cols = m_Src.cols;

    int row, col, nb_elements, i_1, i_2, j_1, j_2;

    Mat m_Src_border(m_Src.rows + 1,m_Src.cols + 1,m_Src.type());
    Mat m_Dst_border(m_Src.rows + 1,m_Src.cols + 1,m_Src.type());

    int border_type = BORDER_REPLICATE, size_border = 1;

    // Calcul du nombre de diagonales dans l'image
    int number_diag = rows + cols - 1;

    // Copie les images en rajoutant une bordure de 0 (pour la gestion des bords)
    copyMakeBorder(m_Src,m_Src_border,size_border,size_border,size_border,size_border,border_type,0);
    copyMakeBorder(m_Dst,m_Dst_border,size_border,size_border,size_border,size_border,border_type,0);

    for(int current_diag = 0; current_diag < number_diag ;current_diag++){
       // Calcul des extrimités des diagonales
        if(current_diag < rows){ i_1 = current_diag; j_1 = 0;}
        else{ j_1 = current_diag - (rows - 1); i_1 = rows - 1;}

        if(current_diag < cols){ j_2 = current_diag; i_2 = 0;}
        else{ i_2 = current_diag - (cols - 1); j_2 = cols - 1;}

        // calcul du nombre d'éléments dans la diagonale
        nb_elements = min(abs(i_1 - i_2),abs(j_1 - j_2));

        // itération sur la diagonale :)
        for(int k = 0; k < nb_elements; k++){
            row = i_1 - k;
            col = j_1 + k;
            for(int chanel = 0; chanel < 3; chanel++){
                m_Dst_border.at<Vec3b>((row + 1),(col + 1))[chanel] = (m_Dst_border.at<Vec3b>((row + 1) - 1,(col + 1))[chanel] +
                m_Dst_border.at<Vec3b>((row + 1),(col + 1) - 1)[chanel] + m_Src_border.at<Vec3b>((row + 1) + 1,(col + 1))[chanel] +
                m_Src_border.at<Vec3b>((row + 1),(col + 1) + 1)[chanel] + m_Src_border.at<Vec3b>((row + 1),(col + 1))[chanel]) / 5;
            }
        }
    }
    // Copy region of interest into the Dst image
    for(int i = 1; i <= m_Src.rows; ++i){
        for(int j = 1; j <= m_Src.cols; ++j){
            // calculate the value of a pixel for each channel following the gaussSeidel method
            for(int chanel = 0; chanel < 3; chanel++){
                    m_Dst.at<Vec3b>(i - 1,j - 1)[chanel] = m_Dst_border.at<Vec3b>(i,j)[chanel];
            }
        }
    }
    return true;
}
bool Diag_Top(const Mat m_Src_border, Mat &m_Dst_border,int rows,int cols){
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
bool Diag_Bot(const Mat m_Src_border, Mat &m_Dst_border,int rows,int cols){
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
bool copyFromBorder(const Mat m_Dst_border, Mat &m_Dst){
    // Copy region of interest(without borders) into the Dst image
    for(int i = 1; i <= m_Dst.rows; ++i){
        for(int j = 1; j <= m_Dst.cols; ++j){
            // calculate the value of a pixel for each channel following the gaussSeidel method
            for(int chanel = 0; chanel < 3; chanel++){
                    m_Dst.at<Vec3b>(i - 1,j - 1)[chanel] = m_Dst_border.at<Vec3b>(i,j)[chanel];
            }
        }
    }
    return true;
}
bool AddGaussSeidelLoop(const Mat m_Src, Mat &m_Dst, int it){

    // Variables Declaration
    int rows = m_Src.rows, cols = m_Src.cols;
    Mat m_Tmp_border(m_Src.rows + 1,m_Src.cols + 1,m_Src.type());
    Mat m_Dst_border(m_Src.rows + 1,m_Src.cols + 1,m_Src.type());
    // timer
    double start_t, end_t;

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
    start_t = omp_get_wtime();
    for(int i = 0; i < it; i++){
        Diag_Top(m_Tmp_border,m_Dst_border,rows,cols);
        Diag_Bot(m_Tmp_border,m_Dst_border,rows,cols);

        m_Tmp_border = m_Dst_border;
    }
    end_t = omp_get_wtime();
    cout << string(100, '-') << endl;
    cout << "| Gauss Seidel Diagonal approch took " << end_t - start_t << " seconds." << endl;
    cout << string(100, '-') << endl;
    // copie de l'image dans la matrice de destination
    copyFromBorder(m_Dst_border, m_Dst);
    return true;
}