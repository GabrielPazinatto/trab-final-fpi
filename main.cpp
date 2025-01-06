#include "opencv2/opencv.hpp"
#include "qcoreapplication.h"
#include <random>
#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
#include <queue>

class Swatch
{
public:
    std::vector<cv::Vec3b*> grayscale_pixels;
    std::vector<cv::Vec3b*> reference_pixels; 
    Swatch()
    {
    }
};

void region_growing(cv::Mat img, std::vector<cv::Vec3b*>& pixels, cv::Point seed, int threshold, int channel = 1){

    const int CONNECTIVITY = 4;
    const int dx[CONNECTIVITY] = {-1, 1, 0, 0};
    const int dy[CONNECTIVITY] = {0, 0, -1, 1};

    pixels.clear();

    cv::Mat visited = cv::Mat::zeros(img.size(), CV_8U);
    int seed_value = img.at<cv::Vec3b>(seed.y, seed.x)[channel];

    std::queue<cv::Point> to_visit;
    to_visit.push(seed);

    visited.at<uchar>(seed.y, seed.x) = 1;

    int i = 0;
    while(!to_visit.empty()){
        cv::Point current = to_visit.front();
        to_visit.pop();

        pixels.push_back(&img.at<cv::Vec3b>(current.y, current.x));
        
        for(int i = 0; i < CONNECTIVITY; i++){
            int new_x = current.x + dx[i];
            int new_y = current.y + dy[i];

            if(!(new_x >= 0 && new_x < img.cols && new_y >= 0 && new_y < img.rows && !visited.at<uchar>(new_y, new_x))){ 
                continue;
            }

            cv::Vec3b pixel = img.at<cv::Vec3b>(new_y, new_x);
            int diff = abs(seed_value - pixel[channel]);

            if(diff <= threshold){
                visited.at<uchar>(new_y, new_x) = 1;
                to_visit.push(cv::Point(new_x, new_y));
            }
        }
    }
}

// Converts an image from BGR to LAB color space
void convert_bgr_to_lab(cv::Mat &img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2Lab);
}

float get_average_luminance(std::vector<cv::Vec3b*> pixels){
    float sum = 0;
    for(cv::Vec3b* pixel : pixels){
        sum += (*pixel)[0];
    }
    return sum / pixels.size();
}

float get_std_dev_luminance(std::vector<cv::Vec3b*> pixels, float average){
    float sum = 0;
    for(cv::Vec3b* pixel : pixels){
        sum += pow((*pixel)[0] - average, 2);
    }
    return sqrt(sum / (pixels.size() - 1));
}

// Gets the index of the sample whose luminance is the closest to the given luminance
// Using binary search could affect the results?
int closest_match_index(int lum, std::vector<cv::Vec3b*> samples)
{
    int left = 0;
    int right = samples.size() - 1;
    int closest_index = 0;
    int closest_distance = 255;

    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        int sample_lum = (*samples[mid])[0];
        int distance = abs(sample_lum - lum);

        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest_index = mid;
        }

        if (sample_lum < lum)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }

    return closest_index;
}

// Matches the luminance of the source image to the destination image
// Similar to histogram matching, but is a linear transformation that uses neighborhood statistics
void remap_luminance(Swatch swatch)
{
    float average_dst_luminance = get_average_luminance(swatch.grayscale_pixels);
    float dst_std_dev = get_std_dev_luminance(swatch.grayscale_pixels, average_dst_luminance);
    float average_src_luminance = get_average_luminance(swatch.reference_pixels);
    float src_std_dev = get_std_dev_luminance(swatch.reference_pixels, average_src_luminance);

    int lum;
    #pragma omp parallel for
    for (int i = 0; i < swatch.reference_pixels.size(); i++)
    { 
        lum = (*swatch.reference_pixels[i])[0];
        (*swatch.reference_pixels[i])[0] = cv::saturate_cast<signed char>((lum - average_src_luminance) * (dst_std_dev / src_std_dev) + average_dst_luminance);
    }
    
}

void match_colors_in_swatch(std::vector<cv::Vec3b*> samples, cv::Mat dst, Swatch swatch){
    for (int i = 0; i < swatch.grayscale_pixels.size(); i++)
    {
        int lum = (*swatch.grayscale_pixels[i])[0];
        int closest_index = closest_match_index(lum, samples);
        (*swatch.grayscale_pixels[i])[1] = (*samples[closest_index])[1];
        (*swatch.grayscale_pixels[i])[2] = (*samples[closest_index])[2];
    }
}

// Gets a jittered sample of the image
// Separates the image into windows and gets a random pixel from each window
void get_jittered_samples(const cv::Mat src, std::vector<cv::Vec3b*>& samples, Swatch swatch, const int window_size = 5)
{
    int x;
    for (int i = 0; i < swatch.reference_pixels.size(); i += window_size)
    {
        if (i + window_size >= swatch.reference_pixels.size())
        {
            x = i + rand() % (swatch.reference_pixels.size() - i);
        }
        else
        {
            x = i + rand() % window_size;
        }

        samples.push_back(swatch.reference_pixels[x]);
    }
    sort(samples.begin(), samples.end(), [](cv::Vec3b* a, cv::Vec3b* b)
        { return (*a)[0] < (*b)[0]; });
}

std::vector<Swatch> swatches;
Swatch *current_swatch = new Swatch();
int threshold = 0;

void mouseCallbackReference(int event, int x, int y, int, void *userdata)
{
    cv::Mat &image = *(cv::Mat*)userdata;
    static cv::Mat image_clone = image.clone();
    static std::vector<cv::Vec3b*> clone_samples;

    if(event == cv::EVENT_LBUTTONDOWN)
    {
        region_growing(image, current_swatch->reference_pixels, cv::Point(x, y), threshold, 0);
        region_growing(image_clone, clone_samples, cv::Point(x, y), threshold, 0);

        std::cout << "Reference pixels: " << current_swatch->reference_pixels.size() << std::endl;

        for(cv::Vec3b* pixel : clone_samples)
        {
            *pixel = {0, 0, 0};
        }
        cv::imshow("Reference Image", image_clone);

        if(current_swatch->reference_pixels.size() > 0 && current_swatch->grayscale_pixels.size() > 0)
        {
            swatches.push_back(*current_swatch);
            current_swatch = new Swatch();
        }
    }
}

void mouseCallbackGrayscale(int event, int x, int y, int, void *userdata)
{
    cv::Mat &image = *(cv::Mat*)userdata;
    static cv::Mat image_clone = image.clone();
    static std::vector<cv::Vec3b*> clone_samples;

    if(event == cv::EVENT_LBUTTONDOWN)
    {
        region_growing(image, current_swatch->grayscale_pixels, cv::Point(x, y), threshold, 0);
        region_growing(image_clone, clone_samples, cv::Point(x, y), threshold, 0);

        std::cout << "Grayscale pixels: " << current_swatch->grayscale_pixels.size() << std::endl;

        for(cv::Vec3b* pixel : clone_samples)
        {
            *pixel = {0, 0, 0};
        }
        cv::imshow("Grayscale Image", image_clone);

        if(current_swatch->reference_pixels.size() > 0 && current_swatch->grayscale_pixels.size() > 0)
        {
            swatches.push_back(*current_swatch);
            current_swatch = new Swatch();
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
        std::cerr << "Usage: " << argv[0] << " <grayscale_image> <reference_img> <window_size>" << std::endl;
        return -1;
    }

    cv::Mat grayscale_img = cv::imread(argv[1]);
    cv::Mat reference_img = cv::imread(argv[2]);

    if (grayscale_img.empty() || reference_img.empty())
    {
        std::cerr << "Error: Could not open or find the images!" << std::endl;
        return -1;
    }

    cv::Mat res = grayscale_img.clone();

    cv::imshow("Grayscale Image", grayscale_img);
    cv::imshow("Reference Image", reference_img);

    cv::Mat grayscale_clone = grayscale_img.clone();
    cv::Mat reference_clone = reference_img.clone();

    cv::setMouseCallback("Reference Image", mouseCallbackReference, &reference_img);
    cv::setMouseCallback("Grayscale Image", mouseCallbackGrayscale, &grayscale_img);
    cv::createTrackbar("Threshold", "Grayscale Image", &threshold, 255);

    while (cv::waitKey(0) != 32)
    {
        continue;
    }

    std::cout << "Swatches: " << swatches.size() << std::endl;

    convert_bgr_to_lab(grayscale_img);
    convert_bgr_to_lab(reference_img);
    convert_bgr_to_lab(res);
    
    int i = 0;
    std::cout << "Remapping luminance..." << std::endl;
    for(Swatch swatch : swatches)
    {
        remap_luminance(swatch);
    }
 
    std::cout << "Getting Samples and Matching Colors..." << std::endl;

    std::vector<cv::Vec3b*> samples;
    for(Swatch swatch : swatches)
    {
        get_jittered_samples(reference_img, samples, swatch, atoi(argv[3]));
        match_colors_in_swatch(samples, res, swatch);
        samples.erase(samples.begin(), samples.end());
    }

    cv::cvtColor(reference_img, reference_img, cv::COLOR_Lab2BGR);
    cv::imshow("Reference Image", reference_img);

    cv::cvtColor(grayscale_img, grayscale_img, cv::COLOR_Lab2BGR);
    cv::imshow("Grayscale Image", grayscale_img);

    app.exec();
    return 0;
}