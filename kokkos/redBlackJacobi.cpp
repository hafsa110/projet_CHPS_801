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

#define DENOISE_ITER 30

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

  Kokkos::initialize(argc, argv);
  {

  typedef Kokkos::LayoutRight  Layout;

  #ifdef KOKKOS_ENABLE_CUDA
  #define MemSpace Kokkos::CudaSpace
  #endif

  #ifndef MemSpace
  #define MemSpace Kokkos::HostSpace
  #endif

  using ExecSpace = MemSpace::execution_space;
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  typedef Kokkos::RangePolicy<ExecSpace> range_policy;

  struct timeval start, end;
  double elapsed_time;

  int rows = img.rows;
  int cols = img.cols;


  // Allocate Matrix on device.
  typedef Kokkos::View<uchar**[3], ExecSpace> ViewMatrix;

  ViewMatrix mColorDenoise("mColorDenoise", rows, cols);
  ViewMatrix::HostMirror h_mColorDenoise = Kokkos::create_mirror_view(mColorDenoise);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      Vec3b pixels = img.at<Vec3b>(i, j);
      for (int c = 0; c < 3; c++) {
        h_mColorDenoise(i, j, c) = pixels.val[c];
      }
    }
  }

  Kokkos::deep_copy(mColorDenoise, h_mColorDenoise);

  gettimeofday(&start, nullptr);

  for (int it = 0; it<DENOISE_ITER; ++it) {
    //mise a jour des points rouges
    Kokkos::parallel_for("red update", range_policy(0, rows), KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < cols; ++j) {
        if((i+j)%2 == 0){
          for(int c = 0; c < 3; c++){
            mColorDenoise(i,j, c) = (i == 0 || j == 0 || i == rows - 1 || j == cols - 1) ? 0.0 : (mColorDenoise(i - 1, j, c) +
              mColorDenoise(i,j - 1, c) + mColorDenoise(i + 1,j, c) +
              mColorDenoise(i,j + 1, c) + mColorDenoise(i, j, c)) / 5;
          } 
        }
      }
    });

    //mise a jour des points noirs 
    Kokkos::parallel_for("black update", range_policy(0, rows), KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < cols; ++j) {
        if((i+j)%2 == 1){
          for(int c = 0; c < 3; c++){
            mColorDenoise(i,j, c) = (i == 0 || j == 0 || i == rows - 1 || j == cols - 1) ? 0.0 : (mColorDenoise(i - 1,j, c) +
              mColorDenoise(i,j - 1, c) + mColorDenoise(i + 1,j, c) +
              mColorDenoise(i,j + 1, c) + mColorDenoise(i, j, c)) / 5;
          } 
        }
      }
    });
  }

  gettimeofday(&end, nullptr);
  elapsed_time = static_cast<double>(end.tv_sec - start.tv_sec) +
                static_cast<double>(end.tv_usec - start.tv_usec) / 1e6;

  Kokkos::deep_copy(h_mColorDenoise, mColorDenoise);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      Vec3b &pixels = img.at<Vec3b>(i,j);
      for (int c = 0; c < 3; ++c) {
        pixels.val[c] = h_mColorDenoise(i, j, c); 
      }
    }
  }

  cout << "| Red Black with Kokkos approch version took " << elapsed_time << " seconds." << endl;

  imwrite("res/redBlack_res_cuda.jpg", img);

  }
  Kokkos::finalize();
/*
  Mat mColorDenoiseSeq(img.size(),img.type());
  img.copyTo(mColorDenoiseSeq);

  gettimeofday(&start, nullptr);

  for(int it = 0; it < DENOISE_ITER; ++it){
    //mise a jour des points rouges
    for(int i = 0; i < rows; ++i){
      for (int j = 0; j < cols; ++j) {
        if((i+j)%2 == 0){
          for(int chanel = 0; chanel < 3; chanel++){
            mColorDenoiseSeq.at<Vec3b>(i,j)[chanel] = (i == 0 || j == 0 || i == rows - 1 || j == cols - 1) ? 0.0 : (mColorDenoiseSeq.at<Vec3b>(i - 1,j)[chanel] +
              mColorDenoiseSeq.at<Vec3b>(i,j - 1)[chanel] + mColorDenoiseSeq.at<Vec3b>(i + 1,j)[chanel] +
              mColorDenoiseSeq.at<Vec3b>(i,j + 1)[chanel] + mColorDenoiseSeq.at<Vec3b>(i, j)[chanel]) / 5;
          } 
        }
      }
    }

    //mise a jour des points noirs 
    for(int i = 0; i < rows; ++i){
      for (int j = 0; j < cols; ++j) {
        if((i+j)%2 == 1){
          for(int chanel = 0; chanel < 3; chanel++){
            mColorDenoiseSeq.at<Vec3b>(i,j)[chanel] = (i == 0 || j == 0 || i == rows - 1 || j == cols - 1) ? 0.0 : (mColorDenoiseSeq.at<Vec3b>(i - 1,j)[chanel] +
              mColorDenoiseSeq.at<Vec3b>(i,j - 1)[chanel] + mColorDenoiseSeq.at<Vec3b>(i + 1,j)[chanel] +
              mColorDenoiseSeq.at<Vec3b>(i,j + 1)[chanel] + mColorDenoiseSeq.at<Vec3b>(i, j)[chanel]) / 5;
          } 
        }
      }
    }
  }

  gettimeofday(&end, nullptr);
  elapsed_time = static_cast<double>(end.tv_sec - start.tv_sec) +
                      static_cast<double>(end.tv_usec - start.tv_usec) / 1e6;

  cout << "| Red Black sequential approch version took " << elapsed_time << " seconds." << endl;

  */
  return 0;
}
