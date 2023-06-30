#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include "yolo-fastestv2.h"
//#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <rs.hpp> // Include RealSense Cross Platform API
#include "cv-helpers.hpp"
//#include "example.hpp"          // Include short list of convenience functions for rendering

const int histSize = 256;

void drawHistogram(cv::Mat &y_hist, cv::Mat &u_hist, cv::Mat &v_hist)
{

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::normalize(y_hist, y_hist, 0, histImage.rows, cv::NORM_MINMAX, -1,
                  cv::Mat());
    cv::normalize(u_hist, u_hist, 0, histImage.rows, cv::NORM_MINMAX, -1,
                  cv::Mat());
    cv::normalize(v_hist, v_hist, 0, histImage.rows, cv::NORM_MINMAX, -1,
                  cv::Mat());

    for (int i = 1; i < histSize; i++)
    {
        cv::line(
            histImage,
            cv::Point(bin_w * (i - 1), hist_h - cvRound(y_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(y_hist.at<float>(i))),
            cv::Scalar(0, 255, 255), 2, 8, 0);
        cv::line(
            histImage,
            cv::Point(bin_w * (i - 1), hist_h - cvRound(u_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(u_hist.at<float>(i))),
            cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(
            histImage,
            cv::Point(bin_w * (i - 1), hist_h - cvRound(v_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(v_hist.at<float>(i))),
            cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    cv::namedWindow("calcHist Demo", cv::WINDOW_AUTOSIZE);
    cv::imshow("calcHist Demo", histImage);
}

int main(int argc, char **argv) {
    cv::Mat src, dst;

    rs2::pipeline pipe;
    pipe.start();
    rs2::frameset data;

    rs2::frame color_frame; // video frame
    cv::Mat color_mat, yuv_mat, hist; //, gray_mat, gray3_mat;

    // cv::VideoCapture cap;
    // if (argc != 2)
    //     cap.open(2);
    // else
    //     cap.open(argv[1]);

    // if (!cap.isOpened())
    // {
    //     std::cerr << "Failed to load webcam/Video ...\n";
    //     return -1;
    // }

    while(1)
    {
        // if (!cap.read(src))
        // {
        //     std::cerr << "Cannot read file\n";
        //     break;
        // }
        // cv::imshow("Src", src);

        data = pipe.wait_for_frames();
        color_frame = data.get_color_frame();

        // Convert RealSense frame to OpenCV matrix:
        color_mat = frame_to_mat(color_frame);

        cv::cvtColor(color_mat, yuv_mat, cv::COLOR_BGR2YUV); //COLOR_RGB2YUV = 83, COLOR_BGR2YUV = 82

        // Separate the three planes
        std::vector<cv::Mat> yuv_planes;
        cv::split(yuv_mat, yuv_planes);

        // Histogram values range from 0 to 256
        float range[] = {0, 256};
        const float *histRange = {range};

        bool uniform = true;
        bool accumulate = false;

        cv::Mat y_hist, u_hist, v_hist;

        // Calculate histogram of the three planes
        cv::calcHist(&yuv_planes[0], 1, 0, cv::Mat(), y_hist, 1, &histSize,
                     &histRange, uniform, accumulate);
        cv::calcHist(&yuv_planes[1], 1, 0, cv::Mat(), u_hist, 1, &histSize,
                     &histRange, uniform, accumulate);
        cv::calcHist(&yuv_planes[2], 1, 0, cv::Mat(), v_hist, 1, &histSize,
                     &histRange, uniform, accumulate);

        drawHistogram(y_hist, u_hist, v_hist);

        // for(int i = 0; i < histSize; ++i){
        //     if(y_hist.at<float>(i) > 125){
        //         std::cout << i << " : " << (int) y_hist.at<float>(i) << ", ";
        // }
        // std::cout <<  std::endl;

        cv::imshow("Camera YUV", yuv_mat);
        cv::imshow("Camera RGB", color_mat);

        if (cv::waitKey(30) == 27)
            break;
    }

    return 0;
}
