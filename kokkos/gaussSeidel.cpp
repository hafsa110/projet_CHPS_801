#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <gaussSeidel.h>

using namespace cv;
using namespace std;

bool applyJacobi(const Mat mSrc, Mat &mDst,double Mean, double StdDev){
  if(mSrc.empty())
  {
      cout<<"[Error]! Input Image Empty!";
      return 0;
  }

  return true;
}


int main( int argc, char* argv[] )
{
  /*
  int N = -1;         // number of rows 2^12
  int M = -1;         // number of columns 2^10
  int S = -1;         // total size 2^22
  int nrepeat = 100;  // number of repeats of the test

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Rows" ) == 0 ) ) {
      N = pow( 2, atoi( argv[ ++i ] ) );
      printf( "  User N is %d\n", N );
    }
    else if ( ( strcmp( argv[ i ], "-M" ) == 0 ) || ( strcmp( argv[ i ], "-Columns" ) == 0 ) ) {
      M = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User M is %d\n", M );
    }
    else if ( ( strcmp( argv[ i ], "-S" ) == 0 ) || ( strcmp( argv[ i ], "-Size" ) == 0 ) ) {
      S = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User S is %d\n", S );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  y^T*A*x Options:\n" );
      printf( "  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n" );
      printf( "  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n" );
      printf( "  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // EXERCISE: Initialize Kokkos runtime.
  //           Include braces to encapsulate code between initialize and finalize calls
  Kokkos::initialize( argc, argv );
  // {

  // For the sake of simplicity in this exercise, we're using std::malloc directly, but
  // later on we'll learn a better way, so generally don't do this in Kokkos programs.
  // Allocate y, x vectors and Matrix A:
  // EXERCISE: For the inpatient only: replace std::malloc with Kokkos::kokkos_malloc<>
  //           This would enable running on GPUs, if KOKKOS_LAMBDA is used instead of [=]
  //           as capture clause for all lambdas. It will be properly introduced later.
  auto y = static_cast<double*>(std::malloc(N * sizeof(double)));
  auto x = static_cast<double*>(std::malloc(M * sizeof(double)));
  auto A = static_cast<double*>(std::malloc(N * M * sizeof(double)));

  // Initialize y vector.
  // EXERCISE: Convert outer loop to Kokkos::parallel_for.
  /*
  for ( int i = 0; i < N; ++i ) {
    y[ i ] = 1;
  }
  */
  /*
  Kokkos::parallel_for(N, 
    [=] (const int64_t i) {
      y[i] = 1;
    }
  );

  // Initialize x vector.
  // EXERCISE: Convert outer loop to Kokkos::parallel_for.
  /*
  for ( int i = 0; i < M; ++i ) {
    x[ i ] = 1;
  }
  *//*
  Kokkos::parallel_for(M, 
    [=] (const int64_t i) {
      x[i] = 1;
    }
  );

  // Initialize A matrix, note 2D indexing computation.
  // EXERCISE: Convert outer loop to Kokkos::parallel_for.
  Kokkos::parallel_for(N, 
    [=] (const int64_t j) {
      for(int i = 0; i < M; ++i) {
        A[ j * M + i ] = 1;
      }
    }
  );
  /*
  for ( int j = 0; j < N; ++j ) {
    for ( int i = 0; i < M; ++i ) {
      A[ j * M + i ] = 1;
    }
  }
  */
/*
  // Timer products.
  Kokkos::Timer timer;
  struct timeval begin, end;

  gettimeofday( &begin, NULL );

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    // Application: <y,Ax> = y^T*A*x
    double result = 0;

    // EXERCISE: Convert outer loop to Kokkos::parallel_reduce.
    
    Kokkos::parallel_reduce("Reduction", N,
      [=] (const int64_t j, double &temp){
        double temp2 = 0;
        for(int i=0; i<M; ++i){
          temp2 += A[j*M+i] * x[i];
        }
        temp += y[j] * temp2;
      },
      result);
    /*
    for ( int j = 0; j < N; ++j ) {
      double temp2 = 0;

      for ( int i = 0; i < M; ++i ) {
        temp2 += A[ j * M + i ] * x[ i ];
      }

      result += y[ j ] * temp2;
    }
    */
/*
    // Output result.
    if ( repeat == ( nrepeat - 1 ) ) {
      printf( "  Computed result for %d x %d is %lf\n", N, M, result );
    }

    const double solution = (double) N * (double) M;

    if ( result != solution ) {
      printf( "  Error: result( %lf ) != solution( %lf )\n", result, solution );
    }
  }

  gettimeofday( &end, NULL );

  // Calculate time.
  double time = timer.seconds();
  //double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
  double Gbytes = 1.0e-9 * double( sizeof(double) * ( M + M * N + N ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

  std::free(A);
  std::free(y);
  std::free(x);

  // EXERCISE: finalize Kokkos runtime
  // }
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

  return 0;
}
