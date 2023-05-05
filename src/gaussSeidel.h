#ifndef GAUSSSEIDEL_H
#define GAUSSSEIDEL_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

bool AddGaussSeidel(const Mat m_Src, Mat &m_Dst, int it);
bool AddGaussSeidel_wave(const Mat m_Src, Mat &m_Dst);
bool AddGaussSeidelLoop(const Mat m_Src, Mat &m_Dst, int it);
bool Diag_Bot(const Mat m_Src_border, Mat &m_Dst_border,int rows,int cols);
bool Diag_Top(const Mat m_Src_border, Mat &m_Dst_border,int rows,int cols);
bool copyFromBorder(const Mat m_Dst_border, Mat &m_Dst);
#endif