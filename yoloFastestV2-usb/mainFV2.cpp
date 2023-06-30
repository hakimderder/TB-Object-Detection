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

// Main avec mesure de la distance

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

static void draw_objects(cv::Mat& cvImg, const std::vector<TargetBox>& boxes,
    rs2::depth_frame depth_frame, rs2::video_frame color_frame, float scale)
{
    // Setup for rs2_project_color_pixel_to_depth_pixel()
    rs2::video_stream_profile depth_profile = depth_frame.get_profile().as<rs2::video_stream_profile>();
    rs2::video_stream_profile color_profile = color_frame.get_profile().as<rs2::video_stream_profile>();

    rs2_intrinsics depth_intrin = depth_profile.get_intrinsics();
    rs2_intrinsics color_intrin = color_profile.get_intrinsics();

    rs2_extrinsics depth2color_extrin = depth_profile.get_extrinsics_to(color_profile);
    rs2_extrinsics color2depth_extrin = color_profile.get_extrinsics_to(depth_profile);

    for(size_t i = 0; i < boxes.size(); ++i) {

        char text[256];
        float mid_x, mid_y;
        TargetBox box = boxes[i];

        // Calculate box center point
        mid_x = (box.x1 + box.x2) / 2.0;
        mid_y = (box.y1 + box.y2) / 2.0;

        float rgb_src_pixel[2] = {mid_x, mid_y};
        float dpt_dst_pixel[2] = {0};

        // Méthode originale mais nul
        // Calcule la moyenne de la distance des pixels dans la boite
        // cv::Rect object(box.x1, box.y1, box.getHeight(), box.getWidth());
        // object = object & cv::Rect(0, 0, depth_mat.cols, depth_mat.rows);
        // distance = mean(depth_mat(object));

        // Convert RGB pixel to depth pixel for get_distance()
        rs2_project_color_pixel_to_depth_pixel(dpt_dst_pixel, (const uint16_t *)(depth_frame.get_data()),
            scale, 0.1f, 2, &depth_intrin, &color_intrin, &color2depth_extrin, &depth2color_extrin, rgb_src_pixel);

        float distance = depth_frame.get_distance(dpt_dst_pixel[0], dpt_dst_pixel[1]);
        // float distance = depth_frame.get_distance(mid_x, mid_y);

        sprintf(text, "%s %.1f%%, Distance : %.2f m", class_names[box.cate+1], box.score * 100, distance);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = box.x1;
        int y = box.y1 - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > cvImg.cols) x = cvImg.cols - label_size.width;

        // Draw label box
        cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        // Write label text
        cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        // Draw detection box
        cv::rectangle (cvImg, cv::Point(box.x1, box.y1),
                       cv::Point(box.x2, box.y2), cv::Scalar(255,0,0));

        // Draw center point (depth measurement)
        cv::circle(cvImg, cv::Point(rgb_src_pixel[0], rgb_src_pixel[1]), 2, cv::Scalar(0, 255, 0));
    }
}

int main(int argc, char** argv)
{
    float f;
    float FPS[16];
    int i, Fcnt=0;
    // Some timing
    std::chrono::steady_clock::time_point Tbegin, Tend;

    // Start streaming from RealSense camera
    rs2::pipeline pipe;
    rs2::pipeline_profile config = pipe.start();

    // rs2::video_stream_profile profile = config.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

    for(i=0;i<16;i++) FPS[i]=0.0;

    yoloF2.init(true); // We use the GPU of the Jetson Nano

    yoloF2.loadModel("yolo-fastestv2-opt.param","yolo-fastestv2-opt.bin");

    rs2::frameset data;
    rs2::frame depth_frame; // depth frame
    rs2::frame color_frame; // video frame
    cv::Mat color_mat;
    // cv::Mat depth_mat;

    rs2::depth_sensor sensor = config.get_device().first<rs2::depth_sensor>();
    float scale = sensor.get_depth_scale();

    std::cout << "Start grabbing, press ESC on live window to terminate" << std::endl;
	while(1){
        Tbegin = std::chrono::steady_clock::now();

        // Wait for the next set of frames
        data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        // data = align_to.process(data);
        color_frame = data.get_color_frame();
        depth_frame = data.get_depth_frame();

        // Convert RealSense frame to OpenCV matrix:
        color_mat = frame_to_mat(color_frame);
        // depth_mat = depth_frame_to_meters(depth_frame);

        std::vector<TargetBox> boxes;
        yoloF2.detection(color_mat, boxes);
        draw_objects(color_mat, boxes, depth_frame, color_frame, scale);

        Tend = std::chrono::steady_clock::now();

        // Calculate frame rate
        f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(color_mat, cv::format("FPS %0.2f", f/16), cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));

        // Show output
        cv::imshow("Jetson Nano", color_mat);
        char esc = cv::waitKey(5);
        if(esc == 27) break;
	}
	cv::destroyAllWindows();
    pipe.stop();
    return 0;
}
