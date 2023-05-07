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

  int rows = img.rows;
  int cols = img.cols;

  Kokkos::initialize(argc, argv);
  {

  typedef Kokkos::LayoutRight  Layout;

  #ifdef KOKKOS_ENABLE_CUDA
  #define MemSpace Kokkos::CudaSpace
  #endif
  #ifdef KOKKOS_ENABLE_HIP
  #define MemSpace Kokkos::Experimental::HIPSpace
  #endif
  #ifdef KOKKOS_ENABLE_OPENMPTARGET
  #define MemSpace Kokkos::OpenMPTargetSpace
  #endif

  #ifndef MemSpace
  #define MemSpace Kokkos::HostSpace
  #endif

  using ExecSpace = MemSpace::execution_space;
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  typedef Kokkos::RangePolicy<ExecSpace>  range_policy;

  // Allocate Matrix on device.
  Mat h_mColorDenoise(img.size(), img.type());
  img.copyTo(h_mColorDenoise);
  int channels = img.channels(); 
  Kokkos::View<cv::Vec3b**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> mColorDenoise(reinterpret_cast<Vec3b*>(h_mColorDenoise.data), img.rows, img.cols);

  // Create host mirrors of device views.

  // Initialize matrix on host.
  Kokkos::deep_copy(mColorDenoise, h_mColorDenoise);

  Kokkos :: parallel_for(DENOISE_ITER, 
    [=] (const int64_t it) {
      //mise a jour des points rouges
      for(int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j) {
          if((i+j)%2 == 0){
            for(int chanel = 0; chanel < 3; chanel++){
              mColorDenoise(i,j)[chanel] = (i == 0 || j == 0 || i == rows - 1 || j == cols - 1) ? 0.0 : (mColorDenoise(i - 1,j)[chanel] +
                mColorDenoise(i,j - 1)[chanel] + mColorDenoise(i + 1,j)[chanel] +
                mColorDenoise(i,j + 1)[chanel] + mColorDenoise(i, j)[chanel]) / 5;
            } 
          }
        }
      }

      //mise a jour des points noirs 
      for(int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j) {
          if((i+j)%2 == 1){
            for(int chanel = 0; chanel < 3; chanel++){
              mColorDenoise(i,j)[chanel] = (i == 0 || j == 0 || i == rows - 1 || j == cols - 1) ? 0.0 : (mColorDenoise(i - 1,j)[chanel] +
                mColorDenoise(i,j - 1)[chanel] + mColorDenoise(i + 1,j)[chanel] +
                mColorDenoise(i,j + 1)[chanel] + mColorDenoise(i, j)[chanel]) / 5;
            } 
          }
        }
      }
    }
  );

  fprintf(stdout, "Writting the output image of size %dx%d...\n", img.rows, img.cols);

  //imwrite("res/redBlack_res.jpg", mColorDenoise);

  }
  Kokkos::finalize();

  return 0;
}
