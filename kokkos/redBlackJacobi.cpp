#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <redBlackJacobi.h>

using namespace cv;
using namespace std;

#define DENOISE_ITER 15

bool applyJacobi(const Mat m_Src, Mat &m_Dst){
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


int main( int argc, char* argv[] )
{
  /*
  Kokkos::initialize(argc, argv);

  std::cout << "Hello, World!" << std::endl;

  Kokkos::finalize();
  */

  CommandLineParser parser(argc, argv,
                               "{@input   |img/lena.jpg|input image}");
  parser.printMessage();

  String imageName = parser.get<String>("@input");
  string image_path = samples::findFile(imageName);
  Mat img = imread(image_path, IMREAD_COLOR);
  if(img.empty())
  {
      std::cout << "Could not read the image: " << image_path << std::endl;
      return 1;
  }

  Mat mColorDenoise(img.size(),img.type());
  img.copyTo(mColorDenoise);

  int rows = mColorDenoise.rows;
  int cols = mColorDenoise.cols;

  Kokkos::initialize( argc, argv );
  {

  for(int it = 0; it < DENOISE_ITER; ++it){
    //mise a jour des points rouges
    for(int i = 0; i < rows; ++i){
      for (int j = 0; j < cols; ++j) {
        if((i+j)%2 == 0){
          for(int chanel = 0; chanel < 3; chanel++){
            mColorDenoise.at<Vec3b>(i,j)[chanel] = (i == 0 || j == 0 || i == rows - 1 || j == cols - 1) ? 0.0 : (mColorDenoise.at<Vec3b>(i - 1,j)[chanel] +
              mColorDenoise.at<Vec3b>(i,j - 1)[chanel] + mColorDenoise.at<Vec3b>(i + 1,j)[chanel] +
              mColorDenoise.at<Vec3b>(i,j + 1)[chanel] + mColorDenoise.at<Vec3b>(i, j)[chanel]) / 5;
          } 
        }
      }
    }

    //mise a jour des points noirs 
    for(int i = 0; i < rows; ++i){
      for (int j = 0; j < cols; ++j) {
        if((i+j)%2 == 1){
          for(int chanel = 0; chanel < 3; chanel++){
            mColorDenoise.at<Vec3b>(i,j)[chanel] = (i == 0 || j == 0 || i == rows - 1 || j == cols - 1) ? 0.0 : (mColorDenoise.at<Vec3b>(i - 1,j)[chanel] +
              mColorDenoise.at<Vec3b>(i,j - 1)[chanel] + mColorDenoise.at<Vec3b>(i + 1,j)[chanel] +
              mColorDenoise.at<Vec3b>(i,j + 1)[chanel] + mColorDenoise.at<Vec3b>(i, j)[chanel]) / 5;
          } 
        }
      }
    }
  }

  fprintf(stdout, "Writting the output image of size %dx%d...\n", img.rows, img.cols);

  imwrite("res/redBlack_res.jpg", mColorDenoise);

  }
  Kokkos::finalize();

  return 0;
}
