#include "opencv2/opencv.hpp"
#include "qcoreapplication.h"
#include <random>
#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>

class Swatch
{
public:
    cv::Point point1_reference{-1, -1};
    cv::Point point2_reference{-1, -1};
    cv::Point point1_grayscale{-1, -1};
    cv::Point point2_grayscale{-1, -1};
    Swatch()
    {
    }
};

// Converts an image from BGR to LAB color space
void convert_bgr_to_lab(cv::Mat &img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2Lab);
}
// Gets the average luminance of a window in an image
float get_average_luminance(const cv::Mat img, const cv::Point point1, const cv::Point point2)
{
    float luminance_sum = 0;
    int start_row = point1.y > point2.y ? point2.y : point1.y;
    int end_row = point2.y > point1.y ? point2.y : point1.y;
    int start_col = point1.x > point2.x ? point2.x : point1.x;
    int end_col = point2.x > point1.x ? point2.x : point1.x;

    for (int i = start_row; i < end_row; i++)
    {
        for (int j = start_col; j < end_col; j++)
        {
            if (i < img.rows && j < img.cols)
            {
                luminance_sum += img.at<cv::Vec3b>(i, j)[0];
            }
        }
    }
    return luminance_sum / ((end_row - start_row) * (end_col - start_col));
}

// Gets the standard deviation of the luminance of a window in an image
float get_luminance_std_dev(const cv::Mat img, const cv::Point point1, const cv::Point point2, const float average_luminance)
{
    float std_dev = 0;
    int start_row = point1.y > point2.y ? point2.y : point1.y;
    int end_row = point2.y > point1.y ? point2.y : point1.y;
    int start_col = point1.x > point2.x ? point2.x : point1.x;
    int end_col = point2.x > point1.x ? point2.x : point1.x;

    for (int i = start_row; i < end_row; i++)
    {
        for (int j = start_col; j < end_col; j++)
        {
            if (i < img.rows && j < img.cols)
            {
                std_dev += pow(img.at<cv::Vec3b>(i, j)[0] - average_luminance, 2);
            }
        }
    }
    return sqrt(std_dev / ((end_row - start_row) * (end_col - start_col) - 1));
}

// Gets the index of the sample whose luminance is the closest to the given luminance
// Using binary search could affect the results?
int closest_match_index(int lum, std::vector<cv::Vec3b> samples)
{

    int closest_index = 0;
    int closest_distance = 255;

    for (int i = 0; i < samples.size(); i++)
    {
        int distance = abs(samples[i][0] - lum);
        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest_index = i;
        }
    }
    return closest_index;
}

// Matches the luminance of the source image to the destination image
// Similar to histogram matching, but is a linear transformation that uses neighborhood statistics
void remap_luminance(const cv::Mat dst, cv::Mat &src, Swatch swatch)
{
    float average_dst_luminance = get_average_luminance(dst, swatch.point1_grayscale, swatch.point2_grayscale);
    float dst_std_dev = get_luminance_std_dev(dst, swatch.point1_grayscale, swatch.point2_grayscale, average_dst_luminance);
    float average_src_luminance = get_average_luminance(src, swatch.point1_reference, swatch.point2_reference);
    float src_std_dev = get_luminance_std_dev(src, swatch.point1_reference, swatch.point2_reference, average_src_luminance);

    int start_row = swatch.point1_reference.y > swatch.point2_reference.y ? swatch.point2_reference.y : swatch.point1_reference.y;
    int end_row = swatch.point2_reference.y > swatch.point1_reference.y ? swatch.point2_reference.y : swatch.point1_reference.y;
    int start_col = swatch.point1_reference.x > swatch.point2_reference.x ? swatch.point2_reference.x : swatch.point1_reference.x;
    int end_col = swatch.point2_reference.x > swatch.point1_reference.x ? swatch.point2_reference.x : swatch.point1_reference.x;

    int lum;

    for (int i = start_row; i < end_row; i++)
    {
        for (int j = start_col; j < end_col; j++)
        { 
            lum = src.at<cv::Vec3b>(i, j)[0];
            src.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<signed char>((lum - average_src_luminance) * (dst_std_dev / src_std_dev) + average_dst_luminance);
        }
    }
}

// effectively matches the colors of the source image to the destination image
void match_colors_in_swatch(std::vector<cv::Vec3b> samples, cv::Mat dst, cv::Point point1, cv::Point point2)
{
    int start_row = point1.y > point2.y ? point2.y : point1.y;
    int end_row = point2.y > point1.y ? point2.y : point1.y;
    int start_col = point1.x > point2.x ? point2.x : point1.x;
    int end_col = point2.x > point1.x ? point2.x : point1.x;

    for (int i = start_row; i < end_row; i++)
    {
        for (int j = start_col; j < end_col; j++)
        {
            int lum = dst.at<cv::Vec3b>(i, j)[0];
            int closest_index = closest_match_index(lum, samples);
            dst.at<cv::Vec3b>(i, j)[1] = samples[closest_index][1];
            dst.at<cv::Vec3b>(i, j)[2] = samples[closest_index][2];
        }
    }
}

// Gets a jittered sample of the image
// Separates the image into windows and gets a random pixel from each window
void get_swatch_jittered_sample(const cv::Mat src, std::vector<cv::Vec3b> &samples, const Swatch swatch, const int window_size = 5)
{
    int start_row = swatch.point1_reference.y > swatch.point2_reference.y ? swatch.point2_reference.y : swatch.point1_reference.y;
    int end_row = swatch.point2_reference.y > swatch.point1_reference.y ? swatch.point2_reference.y : swatch.point1_reference.y;
    int start_col = swatch.point1_reference.x > swatch.point2_reference.x ? swatch.point2_reference.x : swatch.point1_reference.x;
    int end_col = swatch.point2_reference.x > swatch.point1_reference.x ? swatch.point2_reference.x : swatch.point1_reference.x;

    int x, y;
    for (int i = start_row; i < end_row; i += window_size)
    {
        for (int j = start_col; j < end_col; j += window_size)
        {
            if (i + window_size >= end_row)
            {
                x = i + rand() % (end_row - i);
            }
            else
            {
                x = i + rand() % window_size;
            }

            if (j + window_size >= end_col)
            {
                y = j + rand() % (end_col - j);
            }
            else
            {
                y = j + rand() % window_size;
            }

            samples.push_back(src.at<cv::Vec3b>(x, y));
        }
    }
    sort(samples.begin(), samples.end(), [](cv::Vec3b a, cv::Vec3b b)
         { return a[0] < b[0]; });
}

std::vector<Swatch> swatches;
Swatch *current_swatch = new Swatch();

void mouseCallbackReference(int event, int x, int y, int, void *userdata)
{
    cv::Mat &image = *(cv::Mat *)userdata;
    static bool point1_set = false;
    static bool point2_set = false;

    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if (!point1_set)
        {
            current_swatch->point1_reference = cv::Point(x, y);
            point1_set = true;
        }
        else if (!point2_set)
        {
            current_swatch->point2_reference = cv::Point(x, y);
            point2_set = true;

            // Gerar uma cor aleatória
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, 255);
            cv::Scalar randomColor(dist(gen), dist(gen), dist(gen));

            // Colorir os pixels entre os dois pontos
            cv::rectangle(image, current_swatch->point1_reference, current_swatch->point2_reference, randomColor, 2);
            cv::imshow("Reference Image", image);
        }

        if (current_swatch->point1_grayscale != (cv::Point){-1, -1} && current_swatch->point2_grayscale != (cv::Point){-1, -1} && current_swatch->point1_reference != (cv::Point){-1, -1} && current_swatch->point2_reference != (cv::Point){-1, -1})
        {
            swatches.push_back(*current_swatch);
            current_swatch = new Swatch();
        }

        // Resetar os pontos
        if (point1_set && point2_set)
        {
            point1_set = false;
            point2_set = false;
        }
    }
}

void mouseCallbackGrayscale(int event, int x, int y, int, void *userdata)
{
    cv::Mat &image = *(cv::Mat *)userdata;
    static bool point1_set = false;
    static bool point2_set = false;

    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if (!point1_set)
        {
            current_swatch->point1_grayscale = cv::Point(x, y);
            point1_set = true;
        }
        else if (!point2_set)
        {
            current_swatch->point2_grayscale = cv::Point(x, y);
            point2_set = true;

            // Gerar uma cor aleatória
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, 255);
            cv::Scalar randomColor(dist(gen), dist(gen), dist(gen));

            // Colorir os pixels entre os dois pontos
            cv::rectangle(image, current_swatch->point1_grayscale, current_swatch->point2_grayscale, randomColor, 2);
            cv::imshow("Grayscale Image", image);
        }

        if (current_swatch->point1_grayscale != (cv::Point){-1, -1} && current_swatch->point2_grayscale != (cv::Point){-1, -1} && current_swatch->point1_reference != (cv::Point){-1, -1} && current_swatch->point2_reference != (cv::Point){-1, -1})
        {
            swatches.push_back(*current_swatch);
            current_swatch = new Swatch();
        }

        // Resetar os pontos
        if (point1_set && point2_set)
        {
            point1_set = false;
            point2_set = false;
        }
    }
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    cv::namedWindow("Reference Image");
    cv::namedWindow("Grayscale Image");

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <grayscale_image> <reference_image> <window_size>" << std::endl;
        return -1;
    }

    cv::Mat grayscale_img = cv::imread(argv[1]);
    cv::Mat reference_image = cv::imread(argv[2]);

    if (grayscale_img.empty() || reference_image.empty())
    {
        std::cerr << "Error: Could not open or find the images!" << std::endl;
        return -1;
    }

    cv::Mat res = grayscale_img.clone();

    cv::imshow("Grayscale Image", grayscale_img);
    cv::imshow("Reference Image", reference_image);

    cv::Mat grayscale_clone = grayscale_img.clone();
    cv::Mat reference_clone = reference_image.clone();

    cv::setMouseCallback("Reference Image", mouseCallbackReference, &reference_clone);
    cv::setMouseCallback("Grayscale Image", mouseCallbackGrayscale, &grayscale_clone);

    while (cv::waitKey(0) != 32)
    {
        continue;
    }

    std::cout << "Swatches: " << swatches.size() << std::endl;

    for (Swatch s : swatches)
    {
        std::cout << s.point1_grayscale << "," << s.point2_grayscale << "," << s.point1_reference << "," << s.point2_reference << std::endl;
    }

    convert_bgr_to_lab(grayscale_img);
    convert_bgr_to_lab(reference_image);
    convert_bgr_to_lab(res);
    
    std::cout << "Remapping luminance..." << std::endl;
    for(Swatch s : swatches)
    {
        remap_luminance(grayscale_img, reference_image, s);
        std::cout << "loop " << std::endl;
    }
 
    std::cout << "Sampling and matching colors..." << std::endl;
    std::vector<cv::Vec3b> samples;
    for(Swatch s : swatches)
    {
        get_swatch_jittered_sample(reference_image, samples, s, atoi(argv[3]));
        match_colors_in_swatch(samples, res, s.point1_grayscale, s.point2_grayscale);
        samples.erase(samples.begin(), samples.end());
    }

    std::cout << "Matching colors..." << std::endl;

    cv::cvtColor(res, res, cv::COLOR_Lab2BGR);
    cv::imwrite("result.jpg", res);

    cv::imshow("Result", res);

    app.exec();
    return 0;
}