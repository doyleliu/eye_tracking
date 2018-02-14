#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/videoio/videoio_c.h"
#include "opencv2/highgui/highgui_c.h"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

using namespace std;
using namespace cv;

static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
            "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
            "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
            "   [--try-flip]\n"
            "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

string cascadeName = "/usr/local/Cellar/opencv/3.4.0_1/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";

string nestedCascadeName = "/usr/local/Cellar/opencv/3.4.0_1/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";//检测眼睛
CascadeClassifier cascade, nestedCascade;

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
                   CascadeClassifier& nestedCascade,
                   double scale, bool tryflip);
int detectEye(Mat& im, Mat& tpl, Rect& rect)
{
    vector<Rect> faces, eyes;
    // 多尺度人脸检测
    cascade.detectMultiScale(im, faces,
                             1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30,30));
    // 遍历人脸检测结果查找人眼目标
    for (int i = 0; i < faces.size(); i++)
    {
        Mat face = im(faces[i]);
        // 多尺度人眼检测
        nestedCascade.detectMultiScale(face, eyes,
                                       1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(20,20));
        // 人眼检测区域输出
        if (eyes.size())
        {
            rect = eyes[0] + cv::Point(faces[i].x, faces[i].y);
            tpl  = im(rect);
        }
    }
    return eyes.size();
}

void trackEye(Mat& im, Mat& tpl, Rect& rect)
{
    // 人眼位置
    Size pSize(rect.width * 2, rect.height * 2);
    // 矩形区域
    Rect tRect(rect + pSize -
               Point(pSize.width/2, pSize.height/2));
    tRect &= Rect(0, 0, im.cols, im.rows);
    // 匹配模板生成
    Mat tempMat(tRect.width - tpl.rows + 1,
                tRect.height - tpl.cols + 1, CV_32FC1);
    // 模板匹配
    matchTemplate(im(tRect), tpl, tempMat,
                  CV_TM_SQDIFF_NORMED);
    // 计算最小最大值
    double minval, maxval;
    Point minloc, maxloc;
    minMaxLoc(tempMat, &minval, &maxval,
              &minloc, &maxloc);
    // 区域检测判断
    if (minval <= 0.2)
    {
        rect.x = tRect.x + minloc.x;
        rect.y = tRect.y + minloc.y;
    }
    else
        rect.x = rect.y = rect.width = rect.height = 0;
}

//要使用到的两个cascade文件,用于联合检测人脸,opencv自带了多个xml文件，在opencv安装目录下的/source/data/haarcascades/ 文件夹内

int main()
{
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;
    Rect eyeRect;


    char *srcImageFile = "/Users/liudong/Documents/project/c++/untitled/test.jpg";//测试图片
    double scale = 1;
    if (!cascade.load(cascadeName))//载入cascade文件
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }
    if(!nestedCascade.load(nestedCascadeName))
    {
        cerr << "ERROR: Could not load classifier nestedcascade" << endl;
        help();
        return -1;
    }
    image = imread(srcImageFile);


    cvNamedWindow("result", 1);

    cout << "In image read" << endl;
    if (!image.empty())
    {
//        detectAndDraw(image, cascade, nestedCascade, scale, 0);//检测人脸
        detectEye(image, frameCopy, eyeRect);
        // 人眼跟踪
        trackEye(image, frameCopy, eyeRect);
        // 人眼结果绘制
        rectangle(image, eyeRect, CV_RGB(0,255,0));
        imshow("video", image);
        waitKey(0);
    }
    cvDestroyWindow("result");
    return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
                   CascadeClassifier& nestedCascade,
                   double scale, bool tryflip)
{
    int i = 0;
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] = { CV_RGB(0, 0, 255),
                                     CV_RGB(0, 128, 255),
                                     CV_RGB(0, 255, 255),
                                     CV_RGB(0, 255, 0),
                                     CV_RGB(255, 128, 0),
                                     CV_RGB(255, 255, 0),
                                     CV_RGB(255, 0, 0),
                                     CV_RGB(255, 0, 255) };//用于画线
    Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

    cvtColor(img, gray, COLOR_BGR2GRAY);
    resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    t = (double)cvGetTickCount();
    cascade.detectMultiScale(smallImg, faces,
                             1.1, 2, 0
                                     //|CASCADE_FIND_BIGGEST_OBJECT
                                     //|CASCADE_DO_ROUGH_SEARCH
                                     | CASCADE_SCALE_IMAGE
            ,
                             Size(30, 30));
    if (tryflip)
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale(smallImg, faces2,
                                 1.1, 2, 0
                                         //|CASCADE_FIND_BIGGEST_OBJECT
                                         //|CASCADE_DO_ROUGH_SEARCH
                                         | CASCADE_SCALE_IMAGE
                ,
                                 Size(30, 30));
        for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++)
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)cvGetTickCount() - t;
    printf("detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.));
    for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i % 8];
        int radius;

        double aspect_ratio = (double)r->width / r->height;
        if (0.75 < aspect_ratio && aspect_ratio < 1.3)
        {
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            circle(img, center, radius, color, 3, 8, 0);
        }
        else
            rectangle(img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                      cvPoint(cvRound((r->x + r->width - 1)*scale), cvRound((r->y + r->height - 1)*scale)),
                      color, 3, 8, 0);
        if (nestedCascade.empty())
            continue;
        smallImgROI = smallImg(*r);
        nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
                                       1.1, 2, 0
                                               //|CASCADE_FIND_BIGGEST_OBJECT
                                               //|CASCADE_DO_ROUGH_SEARCH
                                               //|CASCADE_DO_CANNY_PRUNING
                                               | CASCADE_SCALE_IMAGE
                ,
                                       Size(30, 30));
        for (vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++)
        {
            center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
            center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
            radius = cvRound((nr->width + nr->height)*0.25*scale);
            circle(img, center, radius, color, 3, 8, 0);
        }
    }
    cv::imshow("result", img);
}

