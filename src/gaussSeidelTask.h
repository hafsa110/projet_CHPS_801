#ifndef GAUSSSEIDELTASK_H
#define GAUSSSEIDELTASK_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;
bool AddGaussSeidelTask(const Mat m_Src, Mat &m_Dst,int it);
bool Task_Diag_Bot(const Mat m_Src_border, Mat &m_Dst_border,int rows,int cols);
bool Task_Diag_Top(const Mat m_Src_border, Mat &m_Dst_border,int rows,int cols);
#endif 