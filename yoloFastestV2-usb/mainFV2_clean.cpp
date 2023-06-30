// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

//modified 12-31-2021 Q-engineering

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

yoloFastestv2 yoloF2;

const char* class_names[] = {
    "background", "person", "bicycle",
    "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
};

// static void draw_objects(cv::Mat& cvImg, const std::vector<TargetBox>& boxes, cv::Mat depth_mat)
static void draw_objects(cv::Mat& cvImg, const std::vector<TargetBox>& boxes,
    rs2::depth_frame depth_frame, rs2::video_frame color_frame, rs2::pipeline_profile selection)
{
    for(size_t i = 0; i < boxes.size(); i++) {
//        std::cout<<boxes[i].x1<<" "<<boxes[i].y1<<" "<<boxes[i].x2<<" "<<boxes[i].y2
//                <<" "<<boxes[i].score<<" "<<boxes[i].cate<<std::endl;
        // for(int channel = 0, channel < depth_mat.channels(), ++channel){
        //     value = img_data[depth_mat.channels() * (depth_mat.cols * mid_x + mid_y) + channel];
        // }

                // cv::Rect object(box.x1, box.y1, box.getHeight(), box.getWidth());
        // object = object & cv::Rect(0, 0, depth_mat.cols, depth_mat.rows);
        // distance = mean(depth_mat(object));

//        distance = depth_frame.get_distance(mid_x, mid_y);
//        float *img_data = (float *)depth_mat.data;

        char text[256];
        // cv::Scalar distance;
        float mid_x, mid_y;
        TargetBox box = boxes[i];

        auto depth_profile = depth_frame.get_profile().as<rs2::video_stream_profile>();
        auto color_profile = color_frame.get_profile().as<rs2::video_stream_profile>();

        auto depth_intrin = depth_profile.get_intrinsics();
        auto color_intrin = color_profile.get_intrinsics();
        auto depth2color_extrin = depth_profile.get_extrinsics_to(color_profile);
        auto color2depth_extrin = color_profile.get_extrinsics_to(depth_profile);

        mid_x = (boxes[i].x1 + boxes[i].x2) / 2.0;
        mid_y = (boxes[i].y1 + boxes[i].y2) / 2.0;

        float rgb_src_pixel[2] = {mid_x, mid_y};
        float dpt_dst_pixel[2] = {0};

        auto sensor = selection.get_device().first<rs2::depth_sensor>();
        auto scale = sensor.get_depth_scale();

        rs2_project_color_pixel_to_depth_pixel(dpt_dst_pixel, (const uint16_t *)(depth_frame.get_data()),
            scale, 0.1f, 2, &depth_intrin, &color_intrin, &color2depth_extrin, &depth2color_extrin, rgb_src_pixel);

        auto distance = depth_frame.get_distance(dpt_dst_pixel[0], dpt_dst_pixel[1]);
//        auto distance = depth_frame.get_distance(mid_x, mid_y);

        // std::cout << std::setprecision(3) << distance[0] << " " << distance[1] << " " <<  distance[2] << " " << distance[3] << std::endl;

        sprintf(text, "%s %.1f%%, Distance : %.2f m", class_names[box.cate+1], box.score * 100, distance);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = box.x1;
        int y = box.y1 - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > cvImg.cols) x = cvImg.cols - label_size.width;

        cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        cv::rectangle (cvImg, cv::Point(box.x1, box.y1),
                       cv::Point(box.x2, box.y2), cv::Scalar(255,0,0));
    }
}

int main(int argc, char** argv)
{
    float f;
    float FPS[16];
    int i,Fcnt=0;
    cv::Mat frame;
    //some timing
    std::chrono::steady_clock::time_point Tbegin, Tend;
    const float WHRatio = 1;

    // Start streaming from Intel RealSense Camera
    rs2::pipeline pipe;
    auto config = pipe.start();

    auto profile = config.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    cv::Size cropSize;
    if (profile.width() / (float)profile.height() > WHRatio) {
        cropSize = cv::Size(static_cast<int>(profile.height() * WHRatio),
                        profile.height());
    } else {
        cropSize = cv::Size(profile.width(),
                        static_cast<int>(profile.width() / WHRatio));
    }

    cv::Rect crop(cv::Point((profile.width() - cropSize.width) / 2,
                    (profile.height() - cropSize.height) / 2),
              cropSize);



    for(i=0;i<16;i++) FPS[i]=0.0;

    yoloF2.init(true); //we use the GPU of the Jetson Nano

    yoloF2.loadModel("yolo-fastestv2-opt.param","yolo-fastestv2-opt.bin");

//    cv::VideoCapture cap("James.mp4");
//    cv::VideoCapture cap(2);
//    if (!cap.isOpened()) {
//        std::cerr << "ERROR: Unable to open the camera" << std::endl;
//        return 0;
//    }

    rs2::frameset data;
//    rs2::depth_frame depth_frame;
//    rs2::video_frame color_frame;
    cv::Mat color_mat;
    cv::Mat depth_mat;

    std::cout << "Start grabbing, press ESC on Live window to terminate" << std::endl;
	while(1){
//        frame=cv::imread("000139.jpg");  //need to refresh frame before dnn class detection
//        cap >> frame;
//        if (frame.empty()) {
//            std::cerr << "ERROR: Unable to grab from the camera" << std::endl;
//            break;
//        }

        Tbegin = std::chrono::steady_clock::now();

        // Wait for the next set of frames
        data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
//        data = align_to.process(data);
        rs2::video_frame color_frame = data.get_color_frame();
        rs2::depth_frame depth_frame = data.get_depth_frame();

        // Convert RealSense frame to OpenCV matrix:
        color_mat = frame_to_mat(color_frame);
        depth_mat = depth_frame_to_meters(depth_frame);

        // Crop both color and depth frames
//        color_mat = color_mat(crop);
//        depth_mat = depth_mat(crop);
//        depth_frame = depth_frame(crop);

        std::vector<TargetBox> boxes;
        yoloF2.detection(color_mat, boxes);
//        draw_objects(color_mat, boxes, depth_mat);
        draw_objects(color_mat, boxes, depth_frame, color_frame, config);

        Tend = std::chrono::steady_clock::now();

        //calculate frame rate
        f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(color_mat, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));

        //show outputstd::cerr << "ERROR: Unable to grab from the camera" << std::endl;
        cv::imshow("Jetson Nano",color_mat);
        //cv::imwrite("test.jpg",frame);
        char esc = cv::waitKey(5);
        if(esc == 27) break;
	}
    return 0;
}
