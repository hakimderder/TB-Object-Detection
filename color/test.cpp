#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include "yolo-fastestv2.h"
//#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <rs.hpp> // Include RealSense Cross Platform API
#include "cv-helpers.hpp"
//#include "example.hpp"          // Include short list of convenience functions for rendering


int main(){
    // Start streaming from RealSense camera
    rs2::pipeline pipe;

    rs2::config c;
//    c.enable_stream(RS2_STREAM_COLOR, RS2_FORMAT_ANY);  // RS2_FORMAT_Y8 RS2_FORMAT_YUYV

//    pipe.start(c);
    pipe.start();

//    rs2::stream_profile s_profile = p_profile.get_stream(RS2_STREAM_COLOR).as<rs2::stream_profile>();

    rs2::frameset data;
    // rs2::frame depth_frame; // depth frame
    rs2::frame color_frame; // video frame
    cv::Mat color_mat, yuv_mat, y_val; //, gray_mat, gray3_mat;

//    const uint8_t *rgb_frame_data;

    std::vector<cv::Mat> yuv_planes;

    while (1) {
        data = pipe.wait_for_frames();
        color_frame = data.get_color_frame();
        // rgb_frame_data = (const uint8_t *) color_frame.get_data();

        // Convert RealSense frame to OpenCV matrix:
        color_mat = frame_to_mat(color_frame);

        //        std::cout << "FPS : " << s_profile.fps() << std::endl;

        cv::cvtColor(color_mat, yuv_mat, cv::COLOR_BGR2YUV); //COLOR_RGB2YUV = 83, COLOR_BGR2YUV = 82

        cv::split(yuv_mat, yuv_planes);

        for (int i = 0; i < 10; ++i)
        {
            //    std::cout << i << "," << j << " : " << (int)yuv_mat.at<uint8_t>(i, j) << " | ";
            std::cout << i << " : " << (int)yuv_mat.at<uint8_t>(i) << ", ";
        }
        std::cout << std::endl;
        //        ++z;

        // Show output
        cv::imshow("Jetson Nano", yuv_mat);
        char esc = cv::waitKey(5);
        if (esc == 27)
            break;
    }
}

//        cv::cvtColor(color_mat, gray_mat, cv::COLOR_BGR2GRAY);
//
//        cv::Mat channels[3] = {gray_mat, gray_mat, gray_mat};
//
//        cv::merge([channels], 3, gray3_mat);
//
//        cv::cvtColor(gray3_mat, yuv_mat, cv::COLOR_BGR2YUV); //COLOR_RGB2YUV = 83, COLOR_BGR2YUV = 82

    //    for (int i = 0; i < yuv_mat.rows; ++i){
    //     //    p = yuv_mat.ptr<uint8_t>(i);
    //        for (int j = 0; j < yuv_mat.cols * yuv_mat.channels(); ++j){
    //            std::cout << i << " : " << (int)yuv_mat[i + yuv_mat.cols + j] << ", ";
    //     //    std::cout << i << " : " << (int)rgb_frame_data[i] << ", ";
    //         //    if ((int)yuv_mat.at<uint8_t>(i, j) > 150){
    //         //        std::cout << i << "," << j << std::endl;
    //         //    }
    //        }
    //    }
    //    std::cout << std::endl;

//        if ((int)yuv_mat.at<uint8_t>(mid_x, mid_y) > 150) {
//            std::cout << i << " : " << (int)yuv_mat.at<uint8_t>(mid_x - 1 ,  mid_y - 1) <<  ", " << (int)yuv_mat.at<uint8_t>(mid_x, mid_y) << ", " << (int)yuv_mat.at<uint8_t>(mid_x + 1, mid_y + 1) << std::endl;
//        }
