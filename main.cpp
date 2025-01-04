#include "opencv2/opencv.hpp"
#include "QCoreApplication"
#include <random>
#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>

class swatch
{
public:
    cv::Point point1_reference{-1, -1};
    cv::Point point2_reference{-1, -1};
    cv::Point point1_grayscale{-1, -1};
    cv::Point point2_grayscale{-1, -1};
    swatch()
    {
    }
};

// Converts an image from BGR to LAB color space
void convert_bgr_to_lab(cv::Mat &img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2Lab);
}

// Gets the average luminance of a window in an image
// Todo:
// - Calculate the average luminance considering the given pixel to be in the center of the window
float get_average_luminance_in_window(const cv::Mat img, const int row, const int col, const int window_size = 5)
{
    float average_luminance = 0;

    for (int i = row; i < row + window_size; i++)
    {
        for (int j = col; j < col + window_size; j++)
        {
            if (i < img.rows && j < img.cols)
            {
                average_luminance += img.at<cv::Vec3b>(i, j)[0];
            }
        }
    }
    return average_luminance / (window_size * window_size);
}

// Gets the standard deviation of the luminance of a window in an image
// Todo:
// - Calculate the standard deviation considering the given pixel to be in the center of the window
float get_luminance_std_dev_in_window(const cv::Mat img, const float average_luminance, const int row, const int col, const int window_size = 5)
{
    float std_dev = 0;

    for (int i = row; i < row + window_size; i++)
    {
        for (int j = col; j < col + window_size; j++)
        {
            if (i < img.rows && j < img.cols)
            {
                std_dev += pow(img.at<cv::Vec3b>(i, j)[0] - average_luminance, 2);
            }
        }
    }
    return sqrt(std_dev / ((window_size * window_size) - 1));
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
void match_luminance(const cv::Mat dst, cv::Mat &src)
{

#pragma omp parallel for collapse(2)
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {

            int lum = src.at<cv::Vec3b>(i, j)[0];
            float average_dst_luminance = get_average_luminance_in_window(dst, i, j);
            float dst_std_dev = get_luminance_std_dev_in_window(dst, average_dst_luminance, i, j);
            float average_src_luminance = get_average_luminance_in_window(src, i, j);
            float src_std_dev = get_luminance_std_dev_in_window(src, average_src_luminance, i, j);

            src.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<signed char>((lum - average_src_luminance) * (dst_std_dev / src_std_dev) + average_dst_luminance);
        }
    }
}

// effectively matches the colors of the source image to the destination image
void match_colors(std::vector<cv::Vec3b> samples, cv::Mat &dst)
{
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            int lum = dst.at<cv::Vec3b>(i, j)[0];
            int closest_index = closest_match_index(lum, samples);
            // std::cout << " " << dst.at<cv::Vec3b>(i,j)  <<  samples[closest_index] << std::endl;
            dst.at<cv::Vec3b>(i, j)[1] = samples[closest_index][1];
            dst.at<cv::Vec3b>(i, j)[2] = samples[closest_index][2];
        }
    }
}

// Gets a jittered sample of the image
// Separates the image into windows and gets a random pixel from each window
void get_jittered_sample(const cv::Mat src, std::vector<cv::Vec3b> &samples, const int window_size = 5)
{
    int x, y;
    for (int i = 0; i < src.rows; i += window_size)
    {
        for (int j = 0; j < src.cols; j += window_size)
        {

            // if: window is not within image boundaries
            // else: window is within image boundaries

            if (i + window_size >= src.rows)
            {
                x = i + rand() % (src.rows - i) - 1;
            }
            else
            {
                x = i + rand() % window_size;
            }

            if (j + window_size >= src.cols)
            {
                y = j + rand() % (src.cols - j) - 1;
            }
            else
            {
                y = j + rand() % window_size;
            }

            // std::cout <<src.at<cv::Vec3b>(x, y) << std::endl;
            samples.push_back(src.at<cv::Vec3b>(x, y));
        }
    }
    sort(samples.begin(), samples.end(), [](cv::Vec3b a, cv::Vec3b b)
         { return a[0] < b[0]; });
}

std::vector<swatch> swatches;
swatch *current_swatch = new swatch();

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
            current_swatch = new swatch();
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
            current_swatch = new swatch();
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
    }

    for (swatch s : swatches)
    {
        std::cout << s.point1_grayscale << "," << s.point2_grayscale << "," << s.point1_reference << "," << s.point2_reference << std::endl;
    }

    convert_bgr_to_lab(grayscale_img);
    convert_bgr_to_lab(reference_image);

    std::cout << "Remapping luminance..." << std::endl;
    match_luminance(grayscale_img, reference_image);

    std::vector<cv::Vec3b> samples;
    std::cout << "Getting jittered sample..." << std::endl;
    get_jittered_sample(reference_image, samples, atoi(argv[3]));
    std::cout << "Sample size: " << samples.size() << std::endl;

    std::cout << "Matching colors..." << std::endl;
    match_colors(samples, res);

    cv::cvtColor(res, res, cv::COLOR_Lab2BGR);
    cv::imwrite("result.jpg", res);

    cv::imshow("Result", res);

    app.exec();
    return 0;
}