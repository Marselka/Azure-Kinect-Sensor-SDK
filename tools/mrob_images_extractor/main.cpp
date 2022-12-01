// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <string>
#include "transformation_helpers.h"
#include "turbojpeg.h"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <chrono>

#include <algorithm>
#include <thread>
#include <functional>
#include <vector>

#include "cmdparser.h"

#define MAX_NUMBER_OF_CAPTURES 259200
#define CAPTURES_IN_BATCH 256

//From https://stackoverflow.com/questions/36246300/parallel-loops-in-c
/// @param[in] nb_elements : size of your for loop
/// @param[in] functor(start, end) :
/// your function processing a sub chunk of the for loop.
/// "start" is the first index to process (included) until the index "end"
/// (excluded)
/// @code
///     for(int i = start; i < end; ++i)
///         computation(i);
/// @endcode
/// @param use_threads : enable / disable threads.
///
///
static
void parallel_for(unsigned nb_elements,
                  std::function<void (int start, int end)> functor,
                  bool use_threads = true)
{
    // -------
    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

    unsigned batch_size = nb_elements / nb_threads;
    unsigned batch_remainder = nb_elements % nb_threads;

    std::vector< std::thread > my_threads(nb_threads);

    if( use_threads )
    {
        // Multithread execution
        for(unsigned i = 0; i < nb_threads; ++i)
        {
            int start = i * batch_size;
            my_threads[i] = std::thread(functor, start, start+batch_size);
        }
    }
    else
    {
        // Single thread execution (for easy debugging)
        for(unsigned i = 0; i < nb_threads; ++i){
            int start = i * batch_size;
            functor( start, start+batch_size );
        }
    }

    // Deform the elements left
    int start = nb_threads * batch_size;
    functor( start, start+batch_remainder);

    // Wait for the other thread to finish their task
    if( use_threads )
        std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
}

//extract(playback, &capture, output_path, transformation, transformed_depth_image, color_image_width_pixels, color_image_height_pixels, distortion, matrix, compression_params);

bool extract(k4a_capture_t capture, const char * output_path, k4a_transformation_t transformation, k4a_image_t transformed_depth_image,
    int color_image_width_pixels, int color_image_height_pixels, int depth_image_width_pixels, int depth_image_height_pixels, 
    cv::Mat distortion, cv::Matx33d matrix, std::vector<int> compression_params, bool undist_project, bool extract_ir_images)
{
    k4a_image_t depth_image = NULL;
    k4a_image_t color_image = NULL;
    k4a_image_t ir_image = NULL;

    uint64_t color_timestamp;
    uint64_t depth_timestamp;
    char color_filename [1024];
    char depth_filename [1024];
    char ir_filename [1024];

    uint8_t * buffer;
    uint32_t size;

    cv::Mat img_array;
    cv::Mat img_array_undistorted;

    // Fetch color and depth frames
    color_image = k4a_capture_get_color_image(capture);
    if (color_image == 0) {
        k4a_capture_release(capture);
        return true;
    }

    depth_image = k4a_capture_get_depth_image(capture);
    if (depth_image == 0) {
        k4a_image_release(color_image);
        k4a_capture_release(capture);
        return true;
    }

    if (extract_ir_images) {
        ir_image = k4a_capture_get_ir_image(capture);
        if (ir_image == 0) {
            k4a_image_release(color_image);
            k4a_image_release(depth_image);
            k4a_capture_release(capture);
            return true;
        }
    }


    color_timestamp = k4a_image_get_device_timestamp_usec(color_image);
    sprintf (color_filename, "%s/color/%012ld.jpg", output_path, color_timestamp);
    depth_timestamp = k4a_image_get_device_timestamp_usec(depth_image);
    sprintf (depth_filename, "%s/depth/%012ld.png", output_path, depth_timestamp);
    if (extract_ir_images) {
        sprintf (ir_filename, "%s/ir/%012ld.png", output_path, depth_timestamp);
    }

    if (!undist_project) {
        FILE * pFile;

        buffer = k4a_image_get_buffer(color_image);
        size = k4a_image_get_size(color_image);

        pFile = fopen (color_filename, "w");

        if (pFile!=NULL)
        {
            fwrite(buffer, 1, size, pFile);
            fclose (pFile);
        }

        buffer = k4a_image_get_buffer(depth_image);

        img_array = cv::Mat(depth_image_height_pixels, depth_image_width_pixels, CV_16UC1, buffer);
        cv::imwrite(depth_filename,  img_array);

        if (extract_ir_images) {
            buffer = k4a_image_get_buffer(ir_image);

            img_array = cv::Mat(depth_image_height_pixels, depth_image_width_pixels, CV_16UC1, buffer);
            cv::imwrite(ir_filename,  img_array);
        }

    }
    else {
        if (K4A_RESULT_SUCCEEDED !=
            k4a_transformation_depth_image_to_color_camera(transformation, depth_image, transformed_depth_image))
        {
            printf("Failed to compute transformed depth image\n");
            return false;
        }
        try {
        buffer = k4a_image_get_buffer(color_image);
        size = k4a_image_get_size(color_image);

        img_array = cv::imdecode(cv::Mat(1, size, CV_8UC1, buffer), cv::IMREAD_UNCHANGED);
        cv::undistort(img_array, img_array_undistorted, matrix, distortion);
        cv::imwrite(color_filename, img_array_undistorted, compression_params);

        buffer = k4a_image_get_buffer(transformed_depth_image);
        size = k4a_image_get_size(transformed_depth_image);

        img_array = cv::Mat(color_image_height_pixels, color_image_width_pixels, CV_16UC1, buffer);
        cv::undistort(img_array, img_array_undistorted, matrix, distortion);
        cv::imwrite(depth_filename,  img_array_undistorted);
        }
        catch (...) {
            printf("Problem catched!\n");
        }
    }

    if (extract_ir_images) {
        if (ir_image != NULL)
        {
            k4a_image_release(ir_image);
        }
    }
    if (depth_image != NULL)
    {
        k4a_image_release(depth_image);
    }
    if (color_image != NULL)
    {
        k4a_image_release(color_image);
    }
    if (capture != NULL)
    {
        k4a_capture_release(capture);
    }

    return true;
}


static int playback(char *input_path, const char * output_path, bool undist_project, bool extract_ir_images)
{
    k4a_playback_t playback = NULL;

    k4a_calibration_t calibration;
    k4a_transformation_t transformation = NULL;
    k4a_calibration_camera_t calib_color;
    k4a_calibration_camera_t calib_depth;
    struct k4a_calibration_intrinsic_parameters_t::_param param;

    k4a_capture_t array_of_captures[CAPTURES_IN_BATCH];
    k4a_image_t array_of_transformed_depth_images[CAPTURES_IN_BATCH];

    k4a_result_t result;
    k4a_stream_result_t stream_result;

    int color_image_width_pixels;
    int color_image_height_pixels;

    int depth_image_width_pixels;
    int depth_image_height_pixels;

    cv::Mat distortion;
    cv::Matx33d matrix;
    cv::Mat img_array;
    cv::Mat img_array_undistorted;
    std::vector<int> compression_params;

    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(96);

    uint32_t i;
    uint32_t number_of_captures = 0;
    bool flag_to_finish = false;
    bool array_of_extraction_results[CAPTURES_IN_BATCH];

    // Open recording
    result = k4a_playback_open(input_path, &playback);
    if (result != K4A_RESULT_SUCCEEDED || playback == NULL)
    {
        printf("Failed to open recording %s\n", input_path);
        return 1;
    }

    if (K4A_RESULT_SUCCEEDED != k4a_playback_get_calibration(playback, &calibration))
    {
        printf("Failed to get calibration\n");
        return 1;
    }

    transformation = k4a_transformation_create(&calibration);
    
    calib_color = calibration.color_camera_calibration;
    color_image_width_pixels = calib_color.resolution_width;
    color_image_height_pixels = calib_color.resolution_height;
    calib_depth = calibration.depth_camera_calibration;
    depth_image_width_pixels = calib_depth.resolution_width;
    depth_image_height_pixels = calib_depth.resolution_height;
    param = calib_color.intrinsics.parameters.param;
    distortion = (cv::Mat_<double>(8,1) << param.k1, param.k2, param.p1, param.p2, param.k3, param.k4, param.k5, param.k6);
    matrix = cv::Matx33d(param.fx, 0.0, param.cx, 0.0, param.fy, param.cy, 0.0, 0.0, 1.0);
    
    // Init depth frame buffer
    for (i = 0; i < CAPTURES_IN_BATCH; i++) {
        array_of_captures[i] = NULL;
        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                                     color_image_width_pixels,
                                                     color_image_height_pixels,
                                                     color_image_width_pixels * (int)sizeof(uint16_t),
                                                     &array_of_transformed_depth_images[i]))
        {
            printf("Failed to create transformed depth image\n");
            return 1;
        }
    }

    for (uint32_t batch = 0; batch < MAX_NUMBER_OF_CAPTURES / CAPTURES_IN_BATCH; batch++) {
        if (flag_to_finish == true) {
            break;
        }
        printf("Batch #%d, ", batch + 1);

        number_of_captures = 0;
        for (i = 0; i < CAPTURES_IN_BATCH; i++) {
            stream_result = k4a_playback_get_next_capture(playback, &array_of_captures[i]);
            if (stream_result != K4A_STREAM_RESULT_SUCCEEDED || array_of_captures[i] == NULL)
            {
                if (stream_result == K4A_STREAM_RESULT_EOF) {
                    flag_to_finish = true;
                }
                else {
                    printf("Failed to fetch frame\n");
                    return 1;
                }
                break;
            }
        }
        number_of_captures = i;
        printf("%d captures, ", number_of_captures);

        //std::this_thread::sleep_for(std::chrono::seconds(10));
        
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        // The core loop
        parallel_for(number_of_captures, [&](int start, int end){ 
            for(int i = start; i < end; ++i) {
                array_of_extraction_results[i] = extract(array_of_captures[i], 
                    output_path, transformation, array_of_transformed_depth_images[i], 
                    color_image_width_pixels, color_image_height_pixels, 
                    depth_image_width_pixels, depth_image_height_pixels, 
                    distortion, matrix, compression_params, undist_project, extract_ir_images);
                if (array_of_extraction_results[i] == false) {
                    printf("Extraction failed\n");
                    // should be proper handling
                }
            }

        });

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        float res = static_cast< float >(number_of_captures) * 1000000.0 / static_cast< float >(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        printf("%f fps.\n", res);
    }

    // Release
    for (i = 0; i < CAPTURES_IN_BATCH; i++) {
      if (array_of_transformed_depth_images[i] != NULL)
      {
          k4a_image_release(array_of_transformed_depth_images[i]);
      }   
    }
    if (playback != NULL)
    {
        k4a_playback_close(playback);
    }
    if (transformation != NULL)
    {
        k4a_transformation_destroy(transformation);
    }
    return 0;
}

int main(int argc, char **argv)
{
    int extraction_mode = 0;
    int extract_ir_images = 0;
    int return_code = 0;

    CmdParser::OptionParser cmd_parser;
    cmd_parser.RegisterOption("-h|--help", "Prints this help", [&]() {
        printf("mrob_images_extractor [options] input.mkv output_path\n");
        //std::cout << "k4arecorder [options] output.mkv" << std::endl << std::endl;
        cmd_parser.PrintOptions();
        exit(0);
    });
    cmd_parser.RegisterOption("--mode",
                              "Specify extraction mode (default: 0).\n"
                              "0 - extraction only\n"
                              "1 - extraction, undistortion, depth to color frame projection.",
                              1,
                              [&](const std::vector<char *> &args) {
                              extraction_mode = std::stoi(args[0]);
                              if (extraction_mode < 0 || extraction_mode > 1) {
                                  std::ostringstream str;
                                  str << "Mode " << extraction_mode <<" is unknown. Must lie in [0,1]";
                                  throw std::runtime_error(str.str());
                                  }
                              });
    cmd_parser.RegisterOption("--extract-ir-images",
                              "Either extract or not infrared images in addition to color and depth ones (default: 0).\n"
                              "Not available if \"--mode\" equals 1 since infrared to color frame projection doesn't work properly.\n"
                              "0 - do not extract infrared images.\n"
                              "1 - extract infrared images.",
                              1,
                              [&](const std::vector<char *> &args) {
                              extract_ir_images = std::stoi(args[0]);
                              if (extract_ir_images != 0 && extract_ir_images != 1) {
                                  std::ostringstream str;
                                  str << "Infrared images extraction mode " << extract_ir_images <<" is unknown. Must be either 0 or 1";
                                  throw std::runtime_error(str.str());
                                  }
                              if (extraction_mode == 1 && extract_ir_images == 1) {
                                  std::ostringstream str;
                                  str << "Extraction of infrared images is not available when \"--mode\" equals 1 "
                                  "since infrared to color frame projection doesn't work properly.";
                                  throw std::runtime_error(str.str());
                                  }
                              });

    int args_left = 0;
    try
    {
        args_left = cmd_parser.ParseCmd(argc, argv);
    }
    catch (CmdParser::ArgumentError &e)
    {
        std::cerr << e.option() << ": " << e.what() << std::endl;
        return 1;
    }
    if (args_left == 2)
    {
        return_code = playback(argv[argc - 2], argv[argc -1], extraction_mode, extract_ir_images);
    }
    else
    {
        printf("mrob_images_extractor [options] input.mkv output_path\n");
        cmd_parser.PrintOptions();
        return 1;
    }

    return return_code;
}
