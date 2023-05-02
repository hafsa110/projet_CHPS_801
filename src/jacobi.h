#ifndef JACOBI_H
#define JACOBI_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

bool AddJacobi(const Mat m_Src, Mat &m_Dst);
#endif