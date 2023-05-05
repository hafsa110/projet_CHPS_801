#ifndef GAUSSSEIDEL_H
#define GAUSSSEIDEL_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

bool AddGaussSeidel(const Mat m_Src, Mat &m_Dst);
bool AddGaussSeidel_wave(const Mat m_Src, Mat &m_Dst);
bool AddGaussSeidel_wave_task(const Mat m_Src, Mat &m_Dst);
#endif