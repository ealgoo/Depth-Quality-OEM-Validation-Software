#include <librealsense2/rs.hpp>
#include <librealsense2/h/rs_types.h>
#include <librealsense2/rsutil.h>
#include <json.hpp>
#include "model-views-IQC.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdarg>
#include <thread>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <map>
#include <sstream>
#include <array>
#include <mutex>
#include <set>
#include <thread>

#include <imgui_internal.h>
#include <stb_image_write.h>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

#ifndef _WIN32
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <X11/Xlib.h>
#endif

#pragma comment(lib, "opengl32.lib")

using namespace rs2;
//using namespace rs400;
using namespace cv;
using namespace nlohmann;

#define TIME_TO_CAPTURE 3
#define TESTING_TIME	6
//#define TESTING_TIME	8
#define MAJOR_VERSION 1
#define MINOR_VERSION 0
#define PATCH_VERSION 9

#define MAX_ANGLE 10

bool gStartCapture = false, gStartTesting = false, gJoinTrigger = false;
region_of_interest gRoi = { -1, -1, -1, -1 };
int gCalculate_sts = -1;
float gOffsetRatio = 0;
#ifdef _WIN32
HANDLE gStopThreadEvent, gStartTestingEvent, gSaveImageEvent, gSaveFinishEvent;
#else
static pthread_cond_t testCon;
static pthread_mutex_t testMtx;
static pthread_cond_t saveImgCon;
static pthread_mutex_t saveImgMtx;
pthread_t testThread;
pthread_t saveImgThread;
unsigned int testSts = 0;
unsigned int saveImgSts = 0;
#define TEST 1
#define SAVEIMG 1
#define ENDTEST 4
#endif

#ifdef _WIN32
void testingThreadFun() {

	HANDLE handle[3];
	handle[0] = gStartTestingEvent;
	handle[1] = gSaveFinishEvent;
	handle[2] = gStopThreadEvent;

	int saveCount = 0;

	while (true)
	{
		auto res = WaitForMultipleObjects(3, handle, FALSE, INFINITE);
		if (res == WAIT_OBJECT_0)
		{
			std::this_thread::sleep_for(std::chrono::seconds(TIME_TO_CAPTURE));
			gStartCapture = true;
			//std::this_thread::sleep_for(std::chrono::seconds(TESTING_TIME - TIME_TO_CAPTURE));
			//gStartTesting = false;
		}
		else if (res == WAIT_OBJECT_0 + 1)
		{
			saveCount++;
			if (saveCount == 4)
			{
				gStartTesting = false;
				saveCount = 0;
			}
		}
		else
		{
			return;
		}
	}
}
#else
void* testingThreadFun(void* context){
	pthread_mutex_lock(&testMtx);
	while(true)
	{
		pthread_cond_wait(&testCon, &testMtx);
		if(testSts & TEST)
		{
			std::this_thread::sleep_for(std::chrono::seconds(TIME_TO_CAPTURE));
			gStartCapture = true;
			std::this_thread::sleep_for(std::chrono::seconds(TESTING_TIME - TIME_TO_CAPTURE));
			gStartTesting = false;
		}
		else if(testSts & ENDTEST)
			break;
		testSts = 0;
	}
	pthread_mutex_unlock(&testMtx);
}
#endif

enum DEV_TYPE {
	RS410,
	RS430,
	COUNT
};

struct test_result {
	enum BLOCK{
		C,
		UR,
		BR,
		COUNT
	};

	float fillRate;
	float accuracy;
	float angle;
	std::string serialNumStr;
	//float meanA;
	//float stdA;
	//float fillRate[BLOCK::COUNT];
	//float mean[BLOCK::COUNT];
	//float std[BLOCK::COUNT];
	std::string csvFileName;
	float rmsSubpixel;
	float rmsFittingPlane;

	void clear() {
		/*for (int i = 0; i < BLOCK::COUNT; i++)
		{
			fillRate[i] = 0;
			mean[i] = 0;
			std[i] = 0;
		}*/
		csvFileName = "";
		serialNumStr = "";
		fillRate = 0.f;
		accuracy = 0.f;
		rmsSubpixel = 0.f;
		rmsFittingPlane = 0.f;
		angle = 0.f;
	}
};

struct IQC_config {
	//float fillRatePassP[3];
	float fillRatePassP;
	float accuracyPassP;
	int ROIPercent;
	//int meanPassP;
	int distance;
	//int minDist;
	//int maxDist;
	//int edgeOffset;
	float rmsSubpixelPassRate;
	float rmsFittingPlanePassRate;
	//bool checkColor;
	bool enablePostProcessing;
}gConfig;

struct stream_format_idx {
	int depthSIdx = -1;
	int infra1SIdx = -1;
	int infra2SIdx = -1;
	int infra1ColorFIdx = -1;
	int infra1Y8FIdx = -1;
	int infra2Y8FIdx = -1;
	int rgbSIdx = -1;
	int rgbColorFIdx = -1;

	void reset() {
		depthSIdx = -1; infra1SIdx = -1; infra2SIdx = -1; infra1ColorFIdx = -1; infra1Y8FIdx = -1; infra2Y8FIdx = -1;  rgbSIdx = -1; rgbColorFIdx = -1;
	}
};

struct angles {
	float angle;
	float angle_x;
	float angle_y;
};

//For RMS use
struct snapshot_metrics
{
	enum {
		ACCURACY,
		SUBPIXEL,
		FITTING_RMS,
		FILLRATE
	};
	int width;
	int height;

	rs2::region_of_interest roi;

	float distance;
	/*float angle;
	float angle_x;
	float angle_y;*/
	angles angles;

	plane p;
	std::array<float3, 4> plane_corners;
	float data[4];
};
//end

bool readConfig() {
	try {
		std::ifstream i("config.json");
		json j;
		i >> j;

		//gConfig.fillRatePassP[test_result::BLOCK::C] = j["config"]["FillRatePassPercentageC"].get<float>();
		//gConfig.fillRatePassP[test_result::BLOCK::UR] = j["config"]["FillRatePassPercentageUR"].get<float>();
		//gConfig.fillRatePassP[test_result::BLOCK::BR] = j["config"]["FillRatePassPercentageBR"].get<float>();
		gConfig.fillRatePassP = j["config"]["FillRatePassPercentage"].get<float>();
		gConfig.accuracyPassP = j["config"]["AccuracyPassPercentage"].get<float>();
		//gConfig.meanPassP = j["config"]["MeanPassPercentage"].get<int>();
		gConfig.distance = j["config"]["TestDistance"].get<int>();
		//gConfig.checkColor = (j["config"]["CheckColor"].get<std::string>() == "False" ? false : true);
		gConfig.enablePostProcessing = (j["config"]["PostProcessing"].get<std::string>() == "False" ? false : true);
		//gConfig.edgeOffset = j["config"]["EdgeOffset"].get<int>();
		gConfig.rmsFittingPlanePassRate = j["config"]["RMSFittingPlanePassRate"].get<float>();
		gConfig.rmsSubpixelPassRate = j["config"]["RMSSubpixelPassRate"].get<float>();
		gConfig.ROIPercent = j["config"]["ROIPercentage"].get<int>();
	}
	catch(std::ifstream::failure e){
		return false;
	}

	return true;
}

void drawTestPicture(video_frame&& vframe, test_result* result)
{
	int actual_w = vframe.get_width(), actual_h = vframe.get_height();
	Mat color24(Size(actual_w, actual_h), CV_8UC3, (void*)vframe.get_data(), Mat::AUTO_STEP);
	char text[100];
	sprintf(text, "Accuracy = %.2f, FillRate = %.2f, subpixelRMS = %.2f, fittingRMS = %.2f", result->accuracy, result->fillRate, result->rmsSubpixel, result->rmsFittingPlane);
	//putText(color24, text, Point(_rect.x, _rect.y), CV_FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));
	//rectangle(color24, Point(_rect.x, _rect.y), Point(_rect.x + _rect.w, _rect.y + _rect.h), Scalar(255, 255, 255), 1);
	putText(color24, text, Point(gRoi.min_x, gRoi.min_y), CV_FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));
	rectangle(color24, Point(gRoi.min_x, gRoi.min_y), Point(gRoi.max_x, gRoi.max_y), Scalar(255, 255, 255), 1);
}

void saveResult(test_result* result)
{
	std::ostringstream fileStream;
	//temporily comment out, waiting for more mature RMS calculation.
	//fileStream << "Center, Mean," << result->mean[test_result::C] << ",FillRate," << result->fillRate[test_result::C] << ",Std," << result->std[test_result::C] << ",RMS," << result->rms << "\n";  
	fileStream << "Serial," << result->serialNumStr << ",Angle," << result->angle << ",Accuracy," << result->accuracy << ",FillRate," << result->fillRate << ",subpixel RMS," << result->rmsSubpixel << ",fitting plane RMS," << result->rmsFittingPlane << "\n";
	std::ofstream save_file(result->csvFileName, std::ofstream::binary);
	save_file.write((char*)fileStream.str().data(), fileStream.str().size());
	save_file.close();
}

/*void drawTestPicture(video_frame&& vframe, rect _rect, test_result* result)
{
	int actual_w = vframe.get_width(), actual_h = vframe.get_height();
	Mat color24(Size(actual_w, actual_h), CV_8UC3, (void*)vframe.get_data(), Mat::AUTO_STEP);
	char text[100];
	sprintf(text, "mean = %.2f, std = %.2f, fillrate = %.3f", result->mean[test_result::C], result->std[test_result::C], result->fillRate[test_result::C]);
	putText(color24, text, Point(_rect.x, _rect.y), CV_FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));
	rectangle(color24, Point(_rect.x, _rect.y), Point(_rect.x + _rect.w, _rect.y + _rect.h), Scalar(255, 255, 255), 1);

	memset(text, 0, 100);
	sprintf(text, "mean = %.2f, std = %.2f, fillrate = %.3f", result->mean[test_result::UR], result->std[test_result::UR], result->fillRate[test_result::UR]);
	putText(color24, text, Point(_rect.x + _rect.w, 10), CV_FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));
	rectangle(color24, Point(_rect.x + _rect.w, 0), Point(actual_w - gConfig.edgeOffset, _rect.y), Scalar(255, 255, 255), 1);

	memset(text, 0, 100);
	sprintf(text, "mean = %.2f, std = %.2f, fillrate = %.3f", result->mean[test_result::BR], result->std[test_result::BR], result->fillRate[test_result::BR]);
	putText(color24, text, Point(_rect.x + _rect.w, _rect.y + _rect.h), CV_FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));
	rectangle(color24, Point(_rect.x + _rect.w, _rect.y + _rect.h), Point(actual_w - gConfig.edgeOffset, _rect.y + (2 * _rect.h)), Scalar(255, 255, 255), 1);
}

void saveResult(test_result* result)
{
	std::ostringstream fileStream;
	//temporily comment out, waiting for more mature RMS calculation.
	//fileStream << "Center, Mean," << result->mean[test_result::C] << ",FillRate," << result->fillRate[test_result::C] << ",Std," << result->std[test_result::C] << ",RMS," << result->rms << "\n";  
	fileStream << "Center, Mean," << result->mean[test_result::C] << ",FillRate," << result->fillRate[test_result::C] << ",Std," << result->std[test_result::C] << "\n";
	fileStream << "Upper Right, Mean," << result->mean[test_result::UR] << ",FillRate," << result->fillRate[test_result::UR] << ",Std," << result->std[test_result::UR] << "\n";  
	fileStream << "Center, Mean," << result->mean[test_result::BR] << ",FillRate," << result->fillRate[test_result::BR] << ",Std," << result->std[test_result::BR] << "\n"; 
	std::ofstream save_file(result->csvFileName, std::ofstream::binary);
	save_file.write((char*)fileStream.str().data(), fileStream.str().size());
	save_file.close();
}*/

#ifndef _WIN32
struct CalResultStrut{
	rs2::frame* frame;
	unsigned short* data;
	test_result* result;
	float baseline_mm;
	rs2_intrinsics intrin;
};

#endif

snapshot_metrics analyze_depth_image(const uint16_t* data, int w, int h, float units, float baseline_mm, rs2_intrinsics* intrin);

#ifdef _WIN32
void calResultThreadFun(frame* _frame, unsigned short* data, test_result* result, float baseline_mm, rs2_intrinsics intrin) {
#else
void* calResultThreadFun(void* ctx){
	CalResultStrut* resultSet = (CalResultStrut*)ctx;
	frame* _frame = resultSet->frame; unsigned short* data = resultSet->data; test_result* result = resultSet->result; float baseline_mm = resultSet->baseline_mm; rs2_intrinsics intrin = resultSet->intrin;
#endif
	//char temp[20];
	video_frame vframe = _frame->as<video_frame>();
	int actual_w = vframe.get_width(), actual_h = vframe.get_height();

	//sprintf(temp, "before analyze image");
	//log(RS2_LOG_SEVERITY_DEBUG, temp);
	auto res = analyze_depth_image(data, actual_w, actual_h, 0.001, baseline_mm, &intrin);
	//sprintf(temp, "after analyze image");
	//log(RS2_LOG_SEVERITY_DEBUG, temp);
	result->fillRate = res.data[snapshot_metrics::FILLRATE];
	result->rmsFittingPlane = res.data[snapshot_metrics::FITTING_RMS];
	result->rmsSubpixel = res.data[snapshot_metrics::SUBPIXEL];
	result->accuracy = res.data[snapshot_metrics::ACCURACY];
	result->angle = res.angles.angle;
	drawTestPicture(_frame->as<video_frame>(), result);
	//sprintf(temp, "before save result");
	//log(RS2_LOG_SEVERITY_DEBUG, temp);
	saveResult(result);
#ifdef _WIN32
	//char temp[20];
	//sprintf(temp, "trigger save event");
	//log(RS2_LOG_SEVERITY_DEBUG, temp);
	SetEvent(gSaveImageEvent);
#else
	pthread_mutex_lock(&saveImgMtx);
	pthread_cond_signal(&saveImgCon);
	pthread_cond_signal(&saveImgCon);
	pthread_cond_signal(&saveImgCon);
	pthread_mutex_unlock(&saveImgMtx);
#endif
	/*video_frame vframe = _frame->as<video_frame>();
	rect fRect; 
	int actual_w = vframe.get_width(), actual_h = vframe.get_height(); int validPixelsCount = 0; unsigned long depthSum = 0; int idx = 0;
	//fRect.x = vframe.get_width() / 3; fRect.y = vframe.get_height() / 3;
	//fRect.w = fRect.x; fRect.h = fRect.y;
	fRect.x = actual_w * (0.5f - 0.5f * gConfig.ROIPercent * 0.01); fRect.y = actual_h * (0.5f - 0.5f * gConfig.ROIPercent * 0.01);
	fRect.w = actual_w * (0.5f + 0.5f * gConfig.ROIPercent * 0.01) - fRect.x; fRect.h = actual_h * (0.5f + 0.5f * gConfig.ROIPercent * 0.01) - fRect.y;
	Mat depth16(Size(vframe.get_width(), vframe.get_height()), CV_16U, (void*)data);
	bool test;
	region_of_interest roi = { fRect.x, fRect.y, fRect.x + fRect.w, fRect.y + fRect.h };

	//temporily comment out, waiting for more mature RMS calculation.
	//auto res = analyze_depth_image(data, actual_w, actual_h, 0.001, roi, baseline_mm, &intrin);
	//result->rms = res.RMS;

	//for center testing
	for (int i = fRect.x; i < fRect.x + fRect.w; i++) // width (col), x
	{
		for (int j = fRect.y; j < fRect.y + fRect.h; j++) // height (row), y
		{
			if (depth16.at<ushort>(j, i) != 0)
			{
				validPixelsCount += 1; // valid depth pixel
				depthSum += depth16.at<ushort>(j, i);
			}
		}
	}

	result->fillRateA = validPixelsCount / (fRect.w*fRect.h);
	if (validPixelsCount != 0)
		result->meanA = depthSum / validPixelsCount;
	else
	{
		result->meanA = -1;//test fail.
#ifdef _WIN32
		return;
#else
		return nullptr;
#endif
	}

	depthSum = 0;
	for (int i = fRect.x; i < fRect.x + fRect.w; i++) // width (col), x
	{
		for (int j = fRect.y; j < fRect.y + fRect.h; j++) // height (row), y
		{
			if (depth16.at<ushort>(j, i) != 0)
			{
				depthSum += pow(depth16.at<ushort>(j, i) - result->meanA, 2.0);
			}
		}
	}

	if (validPixelsCount != 0)
		result->stdA = sqrt(depthSum / validPixelsCount);
	
	drawTestPicture(_frame->as<video_frame>(), fRect, result);
	saveResult(result);
#ifdef _WIN32
	SetEvent(gSaveImageEvent);
#else
	pthread_mutex_lock(&saveImgMtx);
	pthread_cond_signal(&saveImgCon);
	pthread_cond_signal(&saveImgCon);
	pthread_cond_signal(&saveImgCon);
	pthread_mutex_unlock(&saveImgMtx);
#endif
*/
}

/*#ifdef _WIN32
void calResultThreadFun(frame* _frame, unsigned short* data, test_result* result, float baseline_mm, rs2_intrinsics intrin) {
#else
void* calResultThreadFun(void* ctx) {
	CalResultStrut* resultSet = (CalResultStrut*)ctx;
	frame* _frame = resultSet->frame; unsigned short* data = resultSet->data; test_result* result = resultSet->result; float baseline_mm = resultSet->baseline_mm; rs2_intrinsics intrin = resultSet->intrin;
#endif
	video_frame vframe = _frame->as<video_frame>();
	rect fRect;
	int actual_w = vframe.get_width(), actual_h = vframe.get_height(); int validPixelsCount = 0; unsigned long depthSum = 0; int idx = 0;
	fRect.x = vframe.get_width() / 3; fRect.y = vframe.get_height() / 3;
	fRect.w = fRect.x; fRect.h = fRect.y;
	Mat depth16(Size(vframe.get_width(), vframe.get_height()), CV_16U, (void*)data);
	bool test;

	//temporily comment out, waiting for more mature RMS calculation.
	//auto res = analyze_depth_image(data, actual_w, actual_h, 0.001, baseline_mm, &intrin, vframe.get_timestamp());
	//result->rms = res.RMS;

	//for center testing
	for (int i = fRect.x; i < fRect.x + fRect.w; i++) // width (col), x
	{
		for (int j = fRect.y; j < fRect.y + fRect.h; j++) // height (row), y
		{
			if (depth16.at<ushort>(j, i) != 0)
			{
				validPixelsCount += 1; // valid depth pixel
				depthSum += depth16.at<ushort>(j, i);
			}
		}
	}

	result->fillRate[test_result::C] = validPixelsCount / (fRect.w*fRect.h);
	if (validPixelsCount != 0)
		result->mean[test_result::C] = depthSum / validPixelsCount;
	else
	{
		result->mean[test_result::C] = -1;//test fail.
#ifdef _WIN32
		return;
#else
		return nullptr;
#endif
	}

	depthSum = 0;
	for (int i = fRect.x; i < fRect.x + fRect.w; i++) // width (col), x
	{
		for (int j = fRect.y; j < fRect.y + fRect.h; j++) // height (row), y
		{
			if (depth16.at<ushort>(j, i) != 0)
			{
				depthSum += pow(depth16.at<ushort>(j, i) - result->mean[test_result::C], 2.0);
			}
		}
	}

	if (validPixelsCount != 0)
		result->std[test_result::C] = sqrt(depthSum / validPixelsCount);

	//for UR testing
	validPixelsCount = 0;
	depthSum = 0;
	for (int i = fRect.x + fRect.w; i < actual_w - gConfig.edgeOffset; i++) // width (col), x
	{
		for (int j = 0; j < fRect.h; j++) // height (row), y
		{
			if (depth16.at<ushort>(j, i) != 0)
			{
				validPixelsCount += 1; // valid depth pixel
				depthSum += depth16.at<ushort>(j, i);
			}
		}
	}

	result->fillRate[test_result::UR] = validPixelsCount / ((actual_w - (fRect.x + fRect.w) - gConfig.edgeOffset)*fRect.h);
	if (validPixelsCount != 0)
		result->mean[test_result::UR] = depthSum / validPixelsCount;
	else
	{
		result->mean[test_result::UR] = -1;//test fail.
#ifdef _WIN32
		return;
#else
		return nullptr;
#endif
	}

	depthSum = 0;
	for (int i = fRect.x + fRect.w; i < actual_w - gConfig.edgeOffset; i++) // width (col), x
	{
		for (int j = 0; j < fRect.h; j++) // height (row), y
		{
			if (depth16.at<ushort>(j, i) != 0)
			{
				depthSum += pow(depth16.at<ushort>(j, i) - result->mean[test_result::UR], 2.0);
			}
		}
	}

	if (validPixelsCount != 0)
		result->std[test_result::UR] = sqrt(depthSum / validPixelsCount);

	//for BR testing
	validPixelsCount = 0;
	depthSum = 0;
	for (int i = fRect.x + fRect.w; i < actual_w - gConfig.edgeOffset; i++) // width (col), x
	{
		for (int j = fRect.y + fRect.h; j < actual_h; j++) // height (row), y
		{
			if (depth16.at<ushort>(j, i) != 0)
			{
				validPixelsCount += 1; // valid depth pixel
				depthSum += depth16.at<ushort>(j, i);
			}
		}
	}

	result->fillRate[test_result::BR] = validPixelsCount / ((actual_w - (fRect.x + fRect.w) - gConfig.edgeOffset)*fRect.h);
	if (validPixelsCount != 0)
		result->mean[test_result::BR] = depthSum / validPixelsCount;
	else
	{
		result->mean[test_result::BR] = -1;//test fail.
#ifdef _WIN32
		return;
#else
		return nullptr;
#endif
	}

	depthSum = 0;
	for (int i = fRect.x + fRect.w; i < actual_w - gConfig.edgeOffset; i++) // width (col), x
	{
		for (int j = fRect.y + fRect.h; j < actual_h; j++) // height (row), y
		{
			if (depth16.at<ushort>(j, i) != 0)
			{
				depthSum += pow(depth16.at<ushort>(j, i) - result->mean[test_result::BR], 2.0);
			}
		}
	}

	if (validPixelsCount != 0)
		result->std[test_result::BR] = sqrt(depthSum / validPixelsCount);

	drawTestPicture(_frame->as<video_frame>(), fRect, result);
	saveResult(result);
#ifdef _WIN32
	SetEvent(gSaveImageEvent);
#else
	pthread_mutex_lock(&saveImgMtx);
	pthread_cond_signal(&saveImgCon);
	pthread_cond_signal(&saveImgCon);
	pthread_cond_signal(&saveImgCon);
	pthread_mutex_unlock(&saveImgMtx);
#endif
}*/

struct saved_frame_data 
{
	frame* _frame = nullptr;
	std::string filename;
	notifications_model* not_model;
	points* _points = nullptr;

	void clear() {
		if(_frame != nullptr)
			delete _frame;
		if (_points != nullptr)
			_points = nullptr;
			//delete _points;
		not_model = nullptr;
	}
};

std::mutex locker;

#ifdef _WIN32
//void saveFrameThreadFun(saved_frame_data data, points* _points) {
void saveFrameThreadFun(saved_frame_data data) {
#else
void* saveFrameThreadFun(void* _data){
	saved_frame_data data = *(saved_frame_data*)_data;
#endif
	try {
#ifdef _WIN32
		WaitForSingleObject(gSaveImageEvent, INFINITE);
#else
		pthread_mutex_lock(&saveImgMtx);
		pthread_cond_wait(&saveImgCon, &saveImgMtx);
		pthread_mutex_unlock(&saveImgMtx);
#endif
		std::lock_guard<std::mutex> lock(locker);
		/*char temp[50];
		sprintf(temp, "enter thread - %s", data.filename);
		log(RS2_LOG_SEVERITY_DEBUG, temp);
		sprintf(temp, "before locker- %s", data.filename);
		log(RS2_LOG_SEVERITY_DEBUG, temp);

		sprintf(temp, "after locker - %s", data.filename);
		log(RS2_LOG_SEVERITY_DEBUG, temp);*/
		video_frame frame = data._frame->as<video_frame>();
		std::string filenamePng = data.filename + ".png";
		std::string filenamePly = data.filename + ".ply";
		stbi_write_png(filenamePng.data(), frame.get_width(), frame.get_height(), frame.get_bytes_per_pixel(), frame.get_data(), frame.get_width() * frame.get_bytes_per_pixel());
		//sprintf(temp, "after write png - %s", data.filename);
		//log(RS2_LOG_SEVERITY_DEBUG, temp);
		//data.not_model->add_notification({ to_string() << "Snapshot was saved to " << filenamePng.data(),
		//0, RS2_LOG_SEVERITY_INFO,
		//RS2_NOTIFICATION_CATEGORY_UNKNOWN_ERROR});
		SetEvent(gSaveFinishEvent);
		//char temp[50];
		//sprintf(temp, "exit thread - %s", data.filename);
		//log(RS2_LOG_SEVERITY_DEBUG, temp);
		/*if (data._points != nullptr)
		{
			//export_to_ply(filenamePly, *(data.not_model), *data._points, frame);
			export_to_ply(filenamePly, *(data.not_model), *data._points, frame);
		}*/
	}
	catch (const std::exception& e)
	{
		const char* error_message = e.what();
	}
}

struct user_data
{
    GLFWwindow* curr_window = nullptr;
    mouse_info* mouse = nullptr;
    context ctx;
    viewer_model* model = nullptr;
};

bool calTestResult(test_result result)
{
	if (result.fillRate * 100.0f < gConfig.fillRatePassP)
		return false;
	if (abs(result.accuracy) > gConfig.accuracyPassP)
		return false;
	if (result.rmsFittingPlane > gConfig.rmsFittingPlanePassRate)
		return false;
	//remove temporarily, waiting for more information about criteria of subpixel.
	//if (result.rmsSubpixel > gConfig.rmsSubpixelPassRate)
		//return false;
	
	return true;
}

void connectInit(std::string& resultFolder, char* outputFilePath, std::string serialNumStr) {
	char buffer[100];
#ifdef _WIN32
	GetModuleFileNameA(NULL, buffer, 100);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	strcpy(outputFilePath, std::string(buffer).substr(0, pos).c_str());
	resultFolder = std::string(std::string(outputFilePath) + "\\Result");
	CreateDirectoryA(resultFolder.c_str(), NULL);
	resultFolder = resultFolder + "\\" + serialNumStr;
	CreateDirectoryA(resultFolder.c_str(), NULL);
#else
	char szTmp[32];
	sprintf(szTmp, "/proc/%d/exe", getpid());
	int rslt = readlink(szTmp, buffer, 100);
	buffer[rslt] = '\0';
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	strcpy(outputFilePath, std::string(buffer).substr(0, pos).c_str());
	resultFolder = std::string(std::string(outputFilePath) + "/Result");
	mkdir("Result", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	resultFolder = resultFolder + "/" + serialNumStr;
	mkdir(resultFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
}

std::vector<const char*> get_string_pointers(const std::vector<std::string>& vec)
{
	std::vector<const char*> res;
	for (auto&& s : vec) res.push_back(s.c_str());
	return res;
}

void setFontForWindow(ImFont*& selected_font, ImFont* _18Font, ImFont* _14Font)
{
    selected_font = _14Font;
	const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    if(mode->width > 1280 && mode->height > 1024)
		selected_font = _18Font;
}


#pragma region RMS_Cal
plane plane_from_point_and_normal(const rs2::float3& point, const rs2::float3& normal)
{
	return{ normal.x, normal.y, normal.z, -(normal.x*point.x + normal.y*point.y + normal.z*point.z) };
}

plane plane_from_points(const std::vector<rs2::float3> points)
{
	if (points.size() < 3) throw std::runtime_error("Not enough points to calculate plane");

	rs2::float3 sum = { 0,0,0 };
	for (auto point : points) sum = sum + point;

	rs2::float3 centroid = sum / float(points.size());

	double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
	for (auto point : points) {
		rs2::float3 temp = point - centroid;
		xx += temp.x * temp.x;
		xy += temp.x * temp.y;
		xz += temp.x * temp.z;
		yy += temp.y * temp.y;
		yz += temp.y * temp.z;
		zz += temp.z * temp.z;
	}

	double det_x = yy*zz - yz*yz;
	double det_y = xx*zz - xz*xz;
	double det_z = xx*yy - xy*xy;

	double det_max = std::max({ det_x, det_y, det_z });
	if (det_max <= 0) return{ 0, 0, 0, 0 };

	rs2::float3 dir{};
	if (det_max == det_x)
	{
		float a = static_cast<float>((xz*yz - xy*zz) / det_x);
		float b = static_cast<float>((xy*yz - xz*yy) / det_x);
		dir = { 1, a, b };
	}
	else if (det_max == det_y)
	{
		float a = static_cast<float>((yz*xz - xy*zz) / det_y);
		float b = static_cast<float>((xy*xz - yz*xx) / det_y);
		dir = { a, 1, b };
	}
	else
	{
		float a = static_cast<float>((yz*xy - xz*yy) / det_z);
		float b = static_cast<float>((xz*xy - yz*xx) / det_z);
		dir = { a, b, 1 };
	}

	return plane_from_point_and_normal(centroid, dir.normalize());
}

inline double evaluate_pixel(const plane& p, const rs2_intrinsics* intrin, float x, float y, float distance, float3& output)
{
	float pixel[2] = { x, y };
	rs2_deproject_pixel_to_point(&output.x, intrin, pixel, distance);
	return evaluate_plane(p, output);
}

inline float3 approximate_intersection(const plane& p, const rs2_intrinsics* intrin, int x, int y, float min, float max)
{
	float3 point;
	auto _far = evaluate_pixel(p, intrin, x, y, max, point);
	if (fabs(max - min) < 1e-3) return point;
	auto _near = evaluate_pixel(p, intrin, x, y, min, point);
	if (_far*_near > 0) return{ 0, 0, 0 };

	auto avg = (max + min) / 2;
	auto mid = evaluate_pixel(p, intrin, x, y, avg, point);
	if (mid*_near < 0) return approximate_intersection(p, intrin, x, y, min, avg);
	return approximate_intersection(p, intrin, x, y, avg, max);
}

inline float3 approximate_intersection(const plane& p, const rs2_intrinsics* intrin, int x, int y)
{
	return approximate_intersection(p, intrin, x, y, 0.f, 1000.f);
}

//float* calRMS(const std::vector<rs2::float3>& points, rs2::plane p, float baseline_mm, float focal_length_pixels, float plane_fit_to_ground_truth_mm) {
void calRMS(const std::vector<rs2::float3>& points, rs2::plane p, float baseline_mm, float focal_length_pixels, float plane_fit_to_ground_truth_mm, float (&result)[4]) {
	static const float TO_METERS = 0.001f;
	static const float TO_MM = 1000.f;
	static const float TO_PERCENT = 100.f;

	//float* result = new float*(4);

	// Calculate fill rate relative to the ROI
	//char temp[20];
	//sprintf(temp, "before fill test\n");
	//log(RS2_LOG_SEVERITY_DEBUG, temp);
	result[snapshot_metrics::FILLRATE] = points.size() / float((gRoi.max_x - gRoi.min_x)*(gRoi.max_y - gRoi.min_y)) * TO_PERCENT;
	//fill->add_value(fill_rate);
	//sprintf(temp, "after fill test\n");
	//log(RS2_LOG_SEVERITY_DEBUG, temp);

	const float bf_factor = baseline_mm * focal_length_pixels * TO_METERS; // also convert point units from mm to meter

	std::vector<rs2::float3> points_set = points;
	std::vector<float> distances;
	std::vector<float> disparities;
	std::vector<float> gt_errors;

	// Reserve memory for the data
	distances.reserve(points.size());
	disparities.reserve(points.size());
	gt_errors.reserve(points.size());

	// Remove outliers [below 0.5% and above 99.5%)
	std::sort(points_set.begin(), points_set.end(), [](const rs2::float3& a, const rs2::float3& b) { return a.z < b.z; });
	size_t outliers = points_set.size() / 200;
	points_set.erase(points_set.begin(), points_set.begin() + outliers); // crop min 0.5% of the dataset
	points_set.resize(points_set.size() - outliers); // crop max 0.5% of the dataset

													 // Convert Z values into Depth values by aligning the Fitted plane with the Ground Truth (GT) plane
													 // Calculate distance and disparity of Z values to the fitted plane.
													 // Use the aligned fit to calculate GT errors
	for (auto point : points_set)
	{
		// Find distance from point to the reconstructed plane
		auto dist2plane = p.a*point.x + p.b*point.y + p.c*point.z + p.d;
		// Project the point to plane in 3D and find distance to the intersection point
		rs2::float3 plane_intersect = { float(point.x - dist2plane*p.a),
			float(point.y - dist2plane*p.b),
			float(point.z - dist2plane*p.c) };

		// Store distance, disparity and gt- error
		distances.push_back(dist2plane * TO_MM);
		disparities.push_back(bf_factor / point.length() - bf_factor / plane_intersect.length());
		gt_errors.push_back(plane_fit_to_ground_truth_mm - (dist2plane * TO_MM));
	}

	// Show Z accuracy metric only when Ground Truth is available
	//z_accuracy->enable(ground_truth_mm > 0);
	//if (ground_truth_mm)
	//{
	//sprintf(temp, "before accuracy test");
	//log(RS2_LOG_SEVERITY_DEBUG, temp);
		std::sort(begin(gt_errors), end(gt_errors));
		auto gt_median = gt_errors[gt_errors.size() / 2];
		result[snapshot_metrics::ACCURACY] = TO_PERCENT * (gt_median / gConfig.distance);

		///sprintf(temp, "after accuracy test");
		//log(RS2_LOG_SEVERITY_DEBUG, temp);
		//z_accuracy->add_value(accuracy);
	//}

	// Calculate Sub-pixel RMS for Stereo-based Depth sensors
	double total_sq_disparity_diff = 0;
	for (auto disparity : disparities)
	{
		total_sq_disparity_diff += disparity*disparity;
	}
	result[snapshot_metrics::SUBPIXEL] = static_cast<float>(std::sqrt(total_sq_disparity_diff / disparities.size()));
	//sub_pixel_rms_error->add_value(rms_subpixel_val);

	// Calculate Plane Fit RMS  (Spatial Noise) mm, and divide the origin to plane distance to get back the percentage.
	float origin2plane = static_cast<float>(-p.d * 1000);
	double plane_fit_err_sqr_sum = std::inner_product(distances.begin(), distances.end(), distances.begin(), 0.);
	result[snapshot_metrics::FITTING_RMS] = static_cast<float>(std::sqrt(plane_fit_err_sqr_sum / distances.size()))/origin2plane*100;
	//return result;
	//plane_fit_rms_error->add_value(rms_error_val);
	/*const double bf_factor = baseline_mm * focal_length_pixels * 0.001; // also convert point units from meter to mm

																		//std::vector<double> distances; // Calculate the distances of all points in the ROI to the fitted plane
	std::vector<std::pair<double, double> > calc;   // Distances and disparities
	calc.reserve(points.size());        // Calculate the distances of all points in the ROI to the fitted plane

										// Calculate the distance and disparity errors from the point cloud to the fitted plane
	for (auto point : points)
	{
		// Find distance from point to the reconstructed plane
		auto dist2plane = p.a*point.x + p.b*point.y + p.c*point.z + p.d;
		// Project the point to plane in 3D and find distance to the intersection point
		rs2::float3 plane_intersect = { float(point.x - dist2plane*p.a),
			float(point.y - dist2plane*p.b),
			float(point.z - dist2plane*p.c) };

		// Store distance and disparity errors
		calc.emplace_back(std::make_pair(std::fabs(dist2plane) * 1000,
			bf_factor / point.length() - bf_factor / plane_intersect.length()));
	}

	std::sort(calc.begin(), calc.end()); // Filter out the X% of the samples that are further away from the mean
	auto begin = calc.begin(), end = calc.end();
	double total_sq_disparity_diff = 0;
	for (auto itr = begin; itr < end; ++itr)
		total_sq_disparity_diff += (*itr).second*(*itr).second;
	return static_cast<float>(std::sqrt(total_sq_disparity_diff / (points.size())));*/
}

bool calPlane(const uint16_t* data, int w, int h, float units, rs2_intrinsics* intrin, std::array<float3, 4>* corners, angles& angles, std::vector<rs2::float3>* _roi_pixels, plane* _p) {
	//region_of_interest roi = { w / 3, h / 3, w * 2 / 3, h * 2 / 3 };

	std::vector<rs2::float3>* roi_pixels;
	if (_roi_pixels == nullptr)
		roi_pixels = new std::vector<rs2::float3>;
	else
		roi_pixels = _roi_pixels;
	std::array<float3, 4> res = { { -1,-1,-1 } };
	std::mutex m;
#pragma omp parallel for
	for (int y = gRoi.min_y; y < gRoi.max_y; ++y)
		for (int x = gRoi.min_x; x < gRoi.max_x; ++x)
		{
			auto depth_raw = data[y*w + x];

			if (depth_raw)
			{
				// units is float
				float pixel[2] = { float(x), float(y) };
				float point[3];
				auto distance = depth_raw * units;

				rs2_deproject_pixel_to_point(point, intrin, pixel, distance);

				std::lock_guard<std::mutex> lock(m);
				roi_pixels->push_back({ point[0], point[1], point[2] });
			}
		}

	if (roi_pixels->size() < 3) { // Not enough pixels in RoI to fit a plane
		return false;
	}

	plane* p;
	if (_p != nullptr)
		p = _p;
	else
		p = new plane();
	*p = plane_from_points(*roi_pixels);
	
	if (*p == plane{ 0, 0, 0, 0 }) { // The points in RoI don't span a plane
		return false;
	}

	corners->at(0) = approximate_intersection(*p, intrin, gRoi.min_x, gRoi.min_y, 0.f, 1000.f);
	corners->at(1) = approximate_intersection(*p, intrin, gRoi.max_x, gRoi.min_y, 0.f, 1000.f);
	corners->at(2) = approximate_intersection(*p, intrin, gRoi.max_x, gRoi.max_y, 0.f, 1000.f);
	corners->at(3) = approximate_intersection(*p, intrin, gRoi.min_x, gRoi.max_y, 0.f, 1000.f);

	angles.angle = static_cast<float>(std::acos(std::abs(p->c)) / M_PI * 180.);

	// Calculate normal
	auto n = float3{ p->a, p->b, p->c };
	auto cam = float3{ 0.f, 0.f, -1.f };
	auto dot = n * cam;
	auto u = cam - n * dot;

	angles.angle_x = u.x;
	angles.angle_y = u.y;

	if (_p == nullptr)
		delete p;
	if (_roi_pixels == nullptr)
		delete roi_pixels;
	return true;
}

bool calPlane(const uint16_t* data, int w, int h, float units, rs2_intrinsics* intrin, std::array<float3, 4>* corners, angles& angles)
{
	return calPlane(data, w, h, units, intrin, corners, angles, nullptr, nullptr);
}

//snapshot_metrics analyze_depth_image(rs2::video_frame depth_frame, float units, float baseline_mm, rs2_intrinsics* intrin)
snapshot_metrics analyze_depth_image(const uint16_t* data, int w, int h, float units, float baseline_mm, rs2_intrinsics* intrin)
{
	snapshot_metrics result{ w, h, gRoi,{} };
	std::vector<rs2::float3>* roi_pixels =  new std::vector<rs2::float3>();
	std::array<float3, 4>* corners = new std::array<float3, 4>(); 
	plane* p = new plane();

	calPlane(data, w, h, units, intrin, corners, result.angles, roi_pixels, p);

	// Calculate intersection point of the camera's optical axis with the plane fit in camera's CS
	float3 plane_fit_pivot = approximate_intersection(*p, intrin, intrin->ppx, intrin->ppy);
	// Find the distance between the "rectified" fit and the ground truth planes.
	float plane_fit_to_gt_dist_mm = (gConfig.distance > 0.f) ? (plane_fit_pivot.z * 1000 - gConfig.distance) : 0;

	/*char temp[30];
	sprintf(temp, "before calRMS\n");
	log(RS2_LOG_SEVERITY_DEBUG, temp);*/

	//result.data = calRMS(*roi_pixels, *p, baseline_mm, intrin->fx, plane_fit_to_gt_dist_mm, result.data);
	calRMS(*roi_pixels, *p, baseline_mm, intrin->fx, plane_fit_to_gt_dist_mm, result.data);

	/*sprintf(temp, "after calRMS\n");
	log(RS2_LOG_SEVERITY_DEBUG, temp);*/

	result.p = *p;
	result.plane_corners = *corners;

	// Distance of origin (the camera) from the plane is encoded in parameter D of the plane
	result.distance = static_cast<float>(-p->d * 1000);
	// Angle can be calculated from param C
	//result.angle = static_cast<float>(std::acos(std::abs(p->c)) / M_PI * 180.);
	//result.angles.angle = static_cast<float>(std::acos(std::abs(p->c)) / M_PI * 180.);

	// Calculate normal
	/*auto n = float3{ p->a, p->b, p->c };
	auto cam = float3{ 0.f, 0.f, -1.f };
	auto dot = n * cam;
	auto u = cam - n * dot;

	result.angles.angle_x = u.x;
	result.angles.angle_y = u.y;*/

	delete p, roi_pixels, corners;
	return result;
}
#pragma endregion RMS_Cal

DEV_TYPE checkDevType(std::string devName)
{
	int idx = devName.find("RealSense ");
	idx += std::string("RealSense ").size();
	std::string model = devName.substr(idx, 3);
	if (model.compare("410") == 0)
		return DEV_TYPE::RS410;
	else if (model.compare("430") == 0)
		return DEV_TYPE::RS430;
	
	return DEV_TYPE::COUNT;
}

void startStreams(float& baseline, rs2_intrinsics& intrin, stream_format_idx sfIdx, std::shared_ptr<subdevice_model>& depthDev, std::shared_ptr<subdevice_model>& colorDev, viewer_model& viewerModel) {
	if (depthDev->streaming)
		return;
	depthDev->stream_enabled[sfIdx.depthSIdx] = true;
	depthDev->stream_enabled[sfIdx.infra1SIdx] = true;
	depthDev->stream_enabled[sfIdx.infra2SIdx] = true;
	//depthDev->ui.selected_format_id[sfIdx.infra1SIdx] = sfIdx.infra1Y8FIdx;//change the infrared1 format from RGB to Y8
	auto profiles = depthDev->get_selected_profiles();
	auto profilesColor = colorDev->get_selected_profiles();
	//get baseline while test
	stream_profile left_s, right_s;
	for (auto p : profiles)
	{
		if (p.stream_index() == 2 && p.stream_type() == RS2_STREAM_INFRARED)
			right_s = p;
		if (p.stream_index() == 0 && p.stream_type() == RS2_STREAM_DEPTH)
			left_s = p;
	}
	auto extrin = (left_s).get_extrinsics_to(right_s);
	baseline = fabs(extrin.translation[0]) * 1000;  // baseline in mm
	intrin = left_s.as<video_stream_profile>().get_intrinsics();

	depthDev->play(profiles);
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	colorDev->play(profilesColor);

	for (auto&& profile : profiles)
		viewerModel.streams[profile.unique_id()].dev = depthDev;
	for (auto&& profile : profilesColor)
		viewerModel.streams[profile.unique_id()].dev = colorDev;
}

void getAngleSuggestion(angles angles, char* suggestion)
{
	if (angles.angle <= MAX_ANGLE)
	{
		sprintf(suggestion, "%f", angles.angle);
		return;
	}

	if (angles.angle_x > 0.f && fabs(angles.angle_x) > fabs(angles.angle_y))
	{
		sprintf(suggestion, "%s", "Rotate the camera slightly RIGHT");
	}
	else if (angles.angle_x < 0.f && fabs(angles.angle_x) > fabs(angles.angle_y))
	{
		sprintf(suggestion, "%s", "Rotate the camera slightly LEFT");
	}
	else if (angles.angle_y < 0.f && fabs(angles.angle_x) <= fabs(angles.angle_y))
	{
		sprintf(suggestion, "%s", "Rotate the camera slightly UP");
	}
	else if (angles.angle_y > 0.f && fabs(angles.angle_x) <= fabs(angles.angle_y))
	{
		sprintf(suggestion, "%s", "Rotate the camera slightly DOWN");
	}
}

int main(int, char**) try
{
    // Init GUI
    if (!glfwInit()) exit(1);

	//log_to_file(RS2_LOG_SEVERITY_DEBUG, "librealsense.log");


    rs2_error* e = nullptr;
    std::string title = to_string() << "D400 IQC v" << MAJOR_VERSION << "." << MINOR_VERSION << "." << PATCH_VERSION;

    // Create GUI Windows
    auto window = glfwCreateWindow(1280, 1000, title.c_str(), nullptr, nullptr);
    glfwMakeContextCurrent(window);
    ImGui_ImplGlfw_Init(window, true);
    ImFont *font_18, *font_14, *selected_font;
    imgui_easy_theming(font_14, font_18);
	setFontForWindow(selected_font, font_18, font_14);

    std::vector<std::string> restarting_device_info;
    bool isStreaming = false, is_3d_view = false, paused = false, refresh_device_list=true, isPreview = false, showResult = false, nextRoundPlay = false, showTestCondition = false, editable = false/*hasInited = false*/;
    std::vector<std::pair<std::string, std::string>> device_names;
    std::string error_message{ "" }, label{ "" }, resultFolder;
    //std::string ;
    auto last_time_point = std::chrono::high_resolution_clock::now();
    std::vector<device_model> device_models;
    std::vector<device> devs;
	unsigned short* depthImg = nullptr; 
	saved_frame_data frameData[4];
	int saveImgidx = 0, testIdx = 0;
	stream_format_idx sfIdx;
	char buffer[100], outputFilePath[100];
	float baseline_mm = -1.f;
	rs2_intrinsics intrin;

	context ctx;
	device_model* device_to_remove = nullptr;
	viewer_model mViewer_model;
	device_list list;
    std::mutex m;
	DEV_TYPE devType = DEV_TYPE::COUNT;
	test_result testResult;
	angles angles;
    mouse_info mouse;
	int col1 = 20;
	int col2 = 135;
#ifdef _WIN32
	std::thread* saveThread[4] = { nullptr, nullptr, nullptr, nullptr };
	std::thread* calResultThread = nullptr;
#else
	pthread_t saveThread[4] = { NULL, NULL, NULL, NULL };
	pthread_t calResultThread = NULL;
	CalResultStrut testDataSet;
#endif
	

	//get the configuration in JSON file
	readConfig();
	//int deviation = gConfig.distance * gConfig.meanPassP / 100;
	//gConfig.minDist = gConfig.distance - deviation;
	//gConfig.maxDist = gConfig.distance + deviation;

    user_data data;
    data.curr_window = window;
    data.mouse = &mouse;
    data.ctx = ctx;
    data.model = &mViewer_model;

	mViewer_model.enableFilter(gConfig.enablePostProcessing);

    glfwSetWindowUserPointer(window, &data);

    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double cx, double cy)
    {
        auto data = reinterpret_cast<user_data*>(glfwGetWindowUserPointer(w));
        data->mouse->cursor = { (float)cx, (float)cy };
    });
    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int button, int action, int mods)
    {
        auto data = reinterpret_cast<user_data*>(glfwGetWindowUserPointer(w));
        data->mouse->mouse_down = (button == GLFW_MOUSE_BUTTON_1) && (action != GLFW_RELEASE);
    });
    glfwSetScrollCallback(window, [](GLFWwindow * w, double xoffset, double yoffset)
    {
        auto data = reinterpret_cast<user_data*>(glfwGetWindowUserPointer(w));
        data->mouse->mouse_wheel = yoffset;
        data->mouse->ui_wheel += yoffset;
    });

    ctx.set_devices_changed_callback([&](event_information& info)
    {
        std::lock_guard<std::mutex> lock(m);

        for (auto dev : devs)
        {
            if (info.was_removed(dev))
            {
                mViewer_model.not_model.add_notification({ get_device_name(dev).first + " Disconnected\n",
                    0, RS2_LOG_SEVERITY_INFO, RS2_NOTIFICATION_CATEGORY_UNKNOWN_ERROR });
				sfIdx.reset();
				showResult = false;
				gStartTesting = false;
				isPreview = false;
				gCalculate_sts = -1;
				testIdx = 0;
				mViewer_model.selected_depth_source_uid = -1;
				gRoi.min_x = -1;
				isStreaming = false;
            }
        }

        try
        {
            for (auto dev : info.get_new_devices())
            {
                mViewer_model.not_model.add_notification({ get_device_name(dev).first + " Connected\n",
                    0, RS2_LOG_SEVERITY_INFO, RS2_NOTIFICATION_CATEGORY_UNKNOWN_ERROR });
            }
        }
        catch (...)
        {

        }
        refresh_device_list = true;
    });

#ifdef _WIN32
	gStartTestingEvent = CreateEvent(NULL, FALSE, FALSE, TEXT("startTesting"));
	gStopThreadEvent = CreateEvent(NULL, FALSE, FALSE, TEXT("stopTesting"));
	gSaveImageEvent = CreateEvent(NULL, TRUE, FALSE, TEXT("saveImage"));
	gSaveFinishEvent = CreateEvent(NULL, FALSE, FALSE, TEXT("saveImageFinish"));
	std::thread testingThread = std::thread(&testingThreadFun);
#else
	pthread_mutex_init(&testMtx, NULL);
	pthread_mutex_init(&saveImgMtx, NULL);
	pthread_cond_init(&testCon, NULL);
	pthread_cond_init(&saveImgCon, NULL);
	pthread_create(&testThread, NULL, &testingThreadFun, NULL);
#endif

    // Closing the window
    while (!glfwWindowShouldClose(window))
    {
        {
            std::lock_guard<std::mutex> lock(m);

			if (refresh_device_list)
			{
				refresh_device_list = false;

				try
				{
					auto prev_size = list.size();
					list = ctx.query_devices();

					device_names = get_devices_names(list);
					auto dev = [&]() {
						for (size_t i = 0; i < list.size(); i++)
						{
							if (list[i].supports(RS2_CAMERA_INFO_NAME) &&
								std::string(list[i].get_info(RS2_CAMERA_INFO_NAME)) != "Platform Camera")
								return list[i];
						}
						return device();
					}();

					if (dev)
					{
						connectInit(resultFolder, outputFilePath, dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
						testResult.serialNumStr = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
						mViewer_model.not_model.add_log(to_string() << "output folder is : " << resultFolder);
						auto model = device_model(dev, error_message, mViewer_model);
						device_models.push_back(model);
						mViewer_model.not_model.add_log(to_string() << model.dev.get_info(RS2_CAMERA_INFO_NAME) << " was selected as a default device");
					}

					devs.clear();
					for (auto&& sub : list)
					{
						devs.push_back(sub);
						for (auto&& s : sub.query_sensors())
						{
							s.set_notifications_callback([&](const notification& n)
							{
								mViewer_model.not_model.add_notification({ n.get_description(), n.get_timestamp(), n.get_severity(), n.get_category() });
							});
						}
					}


					device_to_remove = nullptr;
					while (true)
					{
						for (auto&& dev_model : device_models)
						{
							bool still_around = false;
							for (auto&& dev : devs)
								if (get_device_name(dev_model.dev) == get_device_name(dev))
									still_around = true;
							if (!still_around) {
								for (auto&& s : dev_model.subdevices)
									s->streaming = false;
								device_to_remove = &dev_model;
							}
						}
						if (device_to_remove)
						{
							device_models.erase(std::find_if(begin(device_models), end(device_models),
								[&](const device_model& other) { return get_device_name(other.dev) == get_device_name(device_to_remove->dev); }));
							device_to_remove = nullptr;
						}
						else break;
					}
				}
				catch (const rs2::error& e)
				{
					error_message = error_to_string(e);
				}
				catch (const std::exception& e)
				{
					error_message = e.what();
				}
			}
        }

        bool update_read_only_options = false;
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(now - last_time_point).count();
        
        if (duration >= 1000)
        {
            update_read_only_options = true;
            last_time_point = now;
        }

#pragma region UI
        glfwPollEvents();
        int w, h;
        glfwGetWindowSize(window, &w, &h);

        const float panel_width = 320.f;
		const float panel_y = 0.f;
        const float default_log_h = 120.f;

        auto output_height = (mViewer_model.is_output_collapsed ? default_log_h : 20);
        ImGui::GetIO().MouseWheel = mouse.ui_wheel;
        mouse.ui_wheel = 0.f;

        ImGui_ImplGlfw_NewFrame(1.0f);

        // Flags for pop-up window - no window resize, move or collaps
        auto flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar|
            ImGuiWindowFlags_NoSavedSettings;

        mViewer_model.show_event_log(selected_font, panel_width,
            h - (mViewer_model.is_output_collapsed ? default_log_h : 20),
            w - panel_width, default_log_h);

        // Set window position and size
        ImGui::SetNextWindowPos({ 10, 0 });
        ImGui::SetNextWindowSize({ panel_width, (float)h });
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0)); 
        ImGui::PushStyleColor(ImGuiCol_WindowBg, from_rgba(0x1b, 0x21, 0x25, 0xff));
        ImGui::Begin("Control Panel", nullptr, flags | ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_AlwaysUseWindowPadding);

        if (device_models.size() > 0)
        {
            std::map<subdevice_model*, float> model_to_y;
            std::map<subdevice_model*, float> model_to_abs_y;
            auto windows_width = ImGui::GetContentRegionMax().x;

            for (auto&& dev_model : device_models)
            {
				auto header_h = 44.f;
                if (dev_model.dev.is<playback>()) header_h += 15;

                ImGui::PushFont(selected_font);
                auto pos = ImGui::GetCursorScreenPos();
                ImGui::GetWindowDrawList()->AddRectFilled(pos, { pos.x + panel_width, pos.y + header_h }, ImColor(sensor_header_light_blue));
                ImGui::GetWindowDrawList()->AddLine({ pos.x,pos.y }, { pos.x + panel_width,pos.y }, ImColor(black));

                pos = ImGui::GetCursorPos();
                ImGui::PushStyleColor(ImGuiCol_Button, sensor_header_light_blue);
                ImGui::SetCursorPos({ 8, pos.y + 14 });
               /* if (dev_model.is_recording)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, redish);
                    label = to_string() << u8"\uf111";
                    ImGui::Text(label.c_str());
                    ImGui::PopStyleColor();
                }
                else if (dev_model.dev.is<playback>())
                {
                    label = to_string() << u8" \uf008";
                    ImGui::Text(label.c_str());
                }
                else
                {*/
                    label = to_string() << u8" \uf03d";
                    ImGui::Text(label.c_str());
                //}
                ImGui::SameLine();

                label = to_string() << dev_model.dev.get_info(RS2_CAMERA_INFO_NAME);
                ImGui::Text(label.c_str());
                ImGui::PushStyleColor(ImGuiCol_Text, from_rgba(0xc3, 0xd5, 0xe5, 0xff));

                /*int playback_control_panel_height = 0;
                if (auto p = dev_model.dev.as<playback>())
                {
                    auto full_path = p.file_name();
                    auto filename = get_file_name(full_path);

                    ImGui::Text("File: \"%s\"", filename.c_str());
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(full_path.c_str());

                    auto playback_panel_pos = ImVec2{ 0, pos.y + panel_y + 18 };
                    ImGui::SetCursorPos(playback_panel_pos);
                    playback_panel_pos.y = dev_model.draw_playback_panel(selected_font, mViewer_model);
                    playback_control_panel_height += playback_panel_pos.y;
                }*/

                ImGui::SetCursorPos({ 0, pos.y + header_h /*+ playback_control_panel_height */});
                pos = ImGui::GetCursorPos();

                int info_control_panel_height = 0;
                if (dev_model.show_device_info)
                {
                    int line_h = 22;
                    info_control_panel_height = dev_model.infos.size() * line_h + 5;

                    const ImVec2 abs_pos = ImGui::GetCursorScreenPos();
                    ImGui::GetWindowDrawList()->AddRectFilled(abs_pos,
                        { abs_pos.x + panel_width, abs_pos.y + info_control_panel_height },
                        ImColor(device_info_color));
                    ImGui::GetWindowDrawList()->AddLine({ abs_pos.x, abs_pos.y - 1 },
                    { abs_pos.x + panel_width, abs_pos.y - 1 },
                        ImColor(black), 1.f);

                    for (auto&& pair : dev_model.infos)
                    {
                        auto rc = ImGui::GetCursorPos();
                        ImGui::SetCursorPos({ rc.x + 12, rc.y + 4 });
                        ImGui::Text("%s:", pair.first.c_str()); ImGui::SameLine();
                        
                        ImGui::PushStyleColor(ImGuiCol_FrameBg, device_info_color);
                        ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, light_blue);
                        ImGui::PushStyleColor(ImGuiCol_Text, white);
                        ImGui::SetCursorPos({ rc.x + 130, rc.y + 1 });
                        label = to_string() << "##" << dev_model.id << " " << pair.first;
                        ImGui::InputText(label.c_str(),
                            (char*)pair.second.data(), 
                            pair.second.size() + 1,
                            ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_ReadOnly);
                        ImGui::PopStyleColor(3);

                        ImGui::SetCursorPos({ rc.x, rc.y + line_h });
                    }
                }

                ImGui::SetCursorPos({ 0, pos.y + info_control_panel_height  });
                ImGui::PopStyleColor(2);
                ImGui::PopFont();

                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, from_rgba(0x1b, 0x21, 0x25, 0xff));
                ImGui::PushStyleColor(ImGuiCol_Text, from_rgba(0xc3, 0xd5, 0xe5, 0xff));
                ImGui::PushFont(selected_font);

				// display device information
				{
					const ImVec2 pos = ImGui::GetCursorPos();
					auto col1 = 135;
					label = to_string() << dev_model.dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION);
					ImGui::Text("FW version:");
					ImGui::SameLine(); ImGui::SetCursorPosX(col1);
					ImGui::Text("%s", label.c_str());

					label = to_string() << api_version_to_string(rs2_get_api_version(&e));
					ImGui::Text("libRS version:");
					ImGui::SameLine(); ImGui::SetCursorPosX(col1);
					ImGui::Text("%s", label.c_str());
					ImGui::SetCursorPos({ ImGui::GetCursorPos().x, ImGui::GetCursorPos().y + 5 });
				}

                // Draw menu foreach subdevice with its properties
                for (auto&& sub : dev_model.subdevices)
                {
                    const ImVec2 pos = ImGui::GetCursorPos();
                    const ImVec2 abs_pos = ImGui::GetCursorScreenPos();
                    model_to_y[sub.get()] = pos.y;
                    model_to_abs_y[sub.get()] = abs_pos.y;
                    ImGui::GetWindowDrawList()->AddLine({ abs_pos.x, abs_pos.y - 1 },
                    { abs_pos.x + panel_width, abs_pos.y - 1 },
                        ImColor(black), 1.f);

                    label = to_string() << sub->s.get_info(RS2_CAMERA_INFO_NAME) << "##" << dev_model.id;
                    ImGui::PushStyleColor(ImGuiCol_Header, sensor_header_light_blue);

                    //ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, { 20, 20 });
                    //ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, { 0, 0 });

					std::string label = to_string() << "Stream Selection Columns##" << sub->dev.get_info(RS2_CAMERA_INFO_NAME)
						<< sub->s.get_info(RS2_CAMERA_INFO_NAME);

					auto col0 = ImGui::GetCursorPosX();
					auto col1 = 135;

					// Draw combo-box with all resolution options for this device
					auto res_chars = get_string_pointers(sub->resolutions);
					ImGui::Text("Resolution:");
					ImGui::SameLine(); ImGui::SetCursorPosX(col1);

					if (!editable)
					{
						ImGui::Text("%s", res_chars[sub->ui.selected_res_id]);
						std::string res = sub->resolutions[sub->ui.selected_res_id];
						auto idx_x = res.find("x");
						int width = stoi(res.substr(0, idx_x-1));
						int height = stoi(res.substr(idx_x+1, res.length()));
						if (gRoi.min_x == -1)
						{
							gRoi = { int(width * (0.5f - 0.5f * gConfig.ROIPercent * 0.01)), int(height * (0.5f - 0.5f * gConfig.ROIPercent * 0.01)),
								int(width * (0.5f + 0.5f * gConfig.ROIPercent * 0.01)), int(height * (0.5f + 0.5f * gConfig.ROIPercent * 0.01)) };
						}
					}
					else
					{
						ImGui::PushItemWidth(-1);
						ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, { 1,1,1,1 });
						ImGui::Combo(label.c_str(), &sub->ui.selected_res_id, res_chars.data(),
							static_cast<int>(res_chars.size()));
						ImGui::PopStyleColor();
						ImGui::PopItemWidth();
					}
					ImGui::SetCursorPosX(col0);

					if (sub->show_single_fps_list)
					{
						auto fps_chars = get_string_pointers(sub->shared_fpses);
						ImGui::Text("Frame Rate:");
						ImGui::SameLine(); ImGui::SetCursorPosX(col1);

						label = to_string() << "##" << sub->dev.get_info(RS2_CAMERA_INFO_NAME)
							<< sub->s.get_info(RS2_CAMERA_INFO_NAME) << " fps";

						if (!editable)
						{
							ImGui::Text("%s", fps_chars[sub->ui.selected_shared_fps_id]);
						}
						else
						{
							ImGui::PushItemWidth(-1);
							ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, { 1,1,1,1 });
							ImGui::Combo(label.c_str(), &sub->ui.selected_shared_fps_id, fps_chars.data(),
								static_cast<int>(fps_chars.size()));
							ImGui::PopStyleColor();
							ImGui::PopItemWidth();
						}

						ImGui::SetCursorPosX(col0);
					}
                    
					ImGui::Text("Streams:");
					for (auto&& f : sub->formats)
					{
						if (sfIdx.depthSIdx == -1)
						{
							if (sub->stream_display_names[f.first] == "Depth")
							{
								sfIdx.depthSIdx = f.first;
								//sub->stream_enabled[sfIdx.depthSIdx] = true;//initial depth is enabled, disable it first.
							}
						}
						int idx = 0;
						if (sfIdx.infra2SIdx == -1)
						{
							if (sub->stream_display_names[f.first] == "Infrared 2")
							{
								sfIdx.infra2SIdx = f.first;
								//sub->stream_enabled[sfIdx.infra2SIdx] = true;
								for (; idx < f.second.size(); idx++)
								{
									if (f.second[idx] == "Y8")
									{
										sfIdx.infra2Y8FIdx = idx;
										sub->ui.selected_format_id[sfIdx.infra2SIdx] = idx;
									}
								}
							}
						}

						idx = 0;
						if (sfIdx.infra1SIdx == -1)
						{
							if (sub->stream_display_names[f.first] == "Infrared 1")
							{
								sfIdx.infra1SIdx = f.first;
								//sub->stream_enabled[sfIdx.infra1SIdx] = true;//initial to enable Infrared 1 stream;
								for (; idx < f.second.size(); idx++)
								{
									if (f.second[idx] == "Y8")
									{
										sfIdx.infra1Y8FIdx = idx;
										sub->ui.selected_format_id[sfIdx.infra1SIdx] = idx;
									}
								}
							}
						}

						if (sfIdx.rgbSIdx == -1)
						{
							if (sub->stream_display_names[f.first] == "Color")
							{
								sfIdx.rgbSIdx = f.first;
								sub->stream_enabled[sfIdx.rgbSIdx] = true;
								for (; idx < f.second.size(); idx++)
								{
									if (f.second[idx] == "RGB8")
										sfIdx.rgbColorFIdx = idx;
								}
							}
						}
					}

					/*for (auto&& f : sub->formats)
					{
						if (f.second.size() == 0)
							continue;

						auto formats_chars = get_string_pointers(f.second);

						if (editable || (!editable && sub->stream_enabled[f.first]))
						{
							if (!editable)
							{
								label = to_string() << "    " << sub->stream_display_names[f.first];
								ImGui::Text("%s", label.c_str());
								
							}
							else
							{
								label = to_string() << sub->stream_display_names[f.first] << "##" << f.first;
								ImGui::Checkbox(label.c_str(), &sub->stream_enabled[f.first]);
							}
						}
					
						if (sub->stream_enabled[f.first])
						{
							if (!editable)
							{
								ImGui::SameLine(); ImGui::SetCursorPosX(col1);
								ImGui::Text("%s", formats_chars[sub->ui.selected_format_id[f.first]]);
							}
							else
							{
								ImGui::PushItemWidth(-1);
								ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, { 1,1,1,1 });
								ImGui::Combo(label.c_str(), &sub->ui.selected_format_id[f.first], formats_chars.data(),
									static_cast<int>(formats_chars.size()));
								ImGui::PopStyleColor();
								ImGui::PopItemWidth();
							}
							ImGui::SetCursorPosX(col0);
						}
						else
						{

						}
					}*/

                    //ImGui::PopStyleVar();
                    //ImGui::PopStyleVar();
                    ImGui::PopStyleColor();
                }

                for (auto&& sub : dev_model.subdevices)
                {
                    sub->update(error_message, mViewer_model.not_model);
                }

                ImGui::PopStyleColor(2);
                ImGui::PopFont();    
            }

            if (device_to_remove)
            {
                if (auto p = device_to_remove->dev.as<playback>())
                {
                    ctx.unload_device(p.file_name());
                }

                device_models.erase(std::find_if(begin(device_models), end(device_models),
                    [&](const device_model& other) { return get_device_name(other.dev) == get_device_name(device_to_remove->dev); }));
                device_to_remove = nullptr;
            }

            //ImGui::SetContentRegionWidth(windows_width);
            auto pos = ImGui::GetCursorScreenPos();
            auto h = ImGui::GetWindowHeight();
            if (h > pos.y - panel_y)
            {
                ImGui::GetWindowDrawList()->AddLine({ pos.x,pos.y }, { pos.x + panel_width,pos.y }, ImColor(from_rgba(0, 0, 0, 0xff)));
                ImRect bb(pos, ImVec2(pos.x + ImGui::GetContentRegionAvail().x, pos.y + ImGui::GetContentRegionAvail().y));
                ImGui::GetWindowDrawList()->AddRectFilled(bb.GetTL(), bb.GetBR(), ImColor(dark_window_background));
            }

			{
				ImGui::SetCursorScreenPos(ImVec2(ImGui::GetCursorScreenPos().x, ImGui::GetCursorScreenPos().y + 10));
				const ImVec2 pos = ImGui::GetCursorPos();
				const ImVec2 abs_pos = ImGui::GetCursorScreenPos();
				float h = 600;
				

				ImGui::PushFont(selected_font);
				ImGui::PushStyleColor(ImGuiCol_Text, from_rgba(0xc3, 0xd5, 0xe5, 0xff));

				ImGui::GetWindowDrawList()->AddLine({ pos.x, pos.y }, { pos.x + panel_width, pos.y }, ImColor(from_rgba(255, 0, 0, 0xff)));
				ImGui::GetWindowDrawList()->AddRectFilled({ abs_pos.x, abs_pos.y }, { abs_pos.x + panel_width, abs_pos.y + h }, ImColor(from_rgba(0, 0, 0, 0xff)));
				/*if (showTestCondition)//default don't show the test criteria.
				{
					ImGui::Text("Pass Condition :");
					ImGui::SetCursorPosX(col1); ImGui::Text("Fill Rate:");
					ImGui::SameLine(); ImGui::SetCursorPosX(col2);
					ImGui::Text("%f, %f, %f", gConfig.fillRatePassP[test_result::BLOCK::C], gConfig.fillRatePassP[test_result::BLOCK::UR], gConfig.fillRatePassP[test_result::BLOCK::BR]);

					ImGui::SetCursorPosX(col1); ImGui::Text("Distance:");
					ImGui::SameLine(); ImGui::SetCursorPosX(col2);
					ImGui::Text("%d", gConfig.distance);

					ImGui::SetCursorPosX(col1); ImGui::Text("Mean Range:");
					ImGui::SameLine(); ImGui::SetCursorPosX(col2);
					ImGui::Text("%d ~ %d", gConfig.minDist, gConfig.maxDist);
				}*/

				if (showResult)
				{
					//ImGui::Text("Test Result "); ImGui::SameLine(); 
					ImGui::Text("Fill Rate:");
					ImGui::SetCursorPosX(col2);
					ImGui::Text("%f", testResult.fillRate);

					ImGui::Text("Accuracy:");
					ImGui::SetCursorPosX(col2);
					ImGui::Text("%f", testResult.accuracy);

					//remove temporarily, waiting for more information about criteria of subpixel.
					/*ImGui::Text("Subpixel RMS (pixel):");
					ImGui::SetCursorPosX(col2);
					ImGui::Text("%f", testResult.rmsSubpixel);*/

					ImGui::Text("Fitting Plane RMS:");
					ImGui::SetCursorPosX(col2);
					ImGui::Text("%f", testResult.rmsFittingPlane);
					/*std::string area[3] = { "Center", "Upper Right", "Bottom Right" };
					ImGui::Text("Test Result:");
					//temporily comment out, waiting for more mature RMS calculation.
					//ImGui::SetCursorPosX(col1); ImGui::Text("Plane RMS:");
					//ImGui::SameLine(); ImGui::SetCursorPosX(col2);
					//ImGui::Text("%f", testResult.rms);

					for (int i = 0; i < test_result::COUNT; i++) {
						label = to_string() << "[" << area[i] << "]";
						ImGui::Text(label.c_str());
						//ImGui::Text("Test Result "); ImGui::SameLine(); 
						ImGui::SetCursorPosX(col1); ImGui::Text("Fill Rate:");
						ImGui::SameLine(); ImGui::SetCursorPosX(col2);
						ImGui::Text("%f", testResult.fillRate[i]);

						ImGui::SetCursorPosX(col1); ImGui::Text("Avg Distance:");
						ImGui::SameLine(); ImGui::SetCursorPosX(col2);
						ImGui::Text("%f", testResult.mean[i]);

						ImGui::SetCursorPosX(col1); ImGui::Text("Std:");
						ImGui::SameLine(); ImGui::SetCursorPosX(col2);
						ImGui::Text("%f", testResult.std[i]);
					}*/
					
				}
			
				ImGui::PopStyleColor(1);
				ImGui::PopFont();

			}

			if(device_models.size() != 0)
            {
                bool stop_recording = false;
				if(device_models[0].subdevices.size() != 0)
                {
					auto subDepth = device_models[0].subdevices[0];
					auto subColor = device_models[0].subdevices[1];
                    try
                    {
                        static float t = 0.f;
                        t += 0.03f;

						ImVec2 pos = ImGui::GetCursorPos();
						ImVec2 abs_pos = ImGui::GetCursorScreenPos();
						float h = 50;

                        ImGui::PushFont(selected_font);
						ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, from_rgba(249,249,29, 255));
                        ImGui::PushStyleColor(ImGuiCol_Button, from_rgba(0x1b + abs(sin(t)) * 40, 0x21 + abs(sin(t)) * 20, 0x25 + abs(sin(t)) * 30, 0xff));
						ImGui::PushStyleColor(ImGuiCol_ButtonHovered, from_rgba(0x1b + abs(sin(t)) * 40, 0x21 + abs(sin(t)) * 20, 0x25 + abs(sin(t)) * 30, 0xff));

						if (!isPreview && !gStartTesting)
						{
							ImGui::SetCursorPosX(abs_pos.x + 25);
							ImGui::SetCursorPosY(abs_pos.y + 10);
							if (ImGui::Button("Preview", { 100, 80 }))
							{
								//enable infrared 1 stream with RGB format for preview
								/*subRS->stream_enabled[sfIdx.infra1SIdx] = true;
								if (sfIdx.infra1ColorFIdx != -1)
									subRS->ui.selected_format_id[sfIdx.infra1SIdx] = sfIdx.infra1ColorFIdx;
								else
									subRS->ui.selected_format_id[sfIdx.infra1SIdx] = sfIdx.infra1Y8FIdx;*/
								startStreams(baseline_mm, intrin, sfIdx, subDepth, subColor, mViewer_model);
								showResult = false;
								isPreview = true;
								/*auto profiles = subRS->get_selected_profiles();
								subRS->play(profiles);
								for (auto&& profile : profiles)
								{
									mViewer_model.streams[profile.unique_id()].dev = subRS;
								}*/
							}
						}
						else {
							ImGui::GetWindowDrawList()->AddRectFilled({ abs_pos.x + 25, abs_pos.y + 10 }, { pos.x + 135, abs_pos.y + 90 }, ImColor(from_rgba(114, 109, 117, 0xff)));
							ImGui::SetCursorPosX(abs_pos.x + 35);
							ImGui::SetCursorPosY(abs_pos.y + 40);
							ImGui::TextDisabled("Preview");
						}

						if (gStartTesting)
						{
							ImGui::GetWindowDrawList()->AddRectFilled({ abs_pos.x + 160, abs_pos.y + 10 }, { pos.x + 270, abs_pos.y + 90 }, ImColor(from_rgba(114, 109, 117, 0xff)));
							ImGui::SetCursorPosX(abs_pos.x + 185);
							ImGui::SetCursorPosY(abs_pos.y + 40);
							ImGui::TextDisabled("Test");
						}
						else
						{
							ImGui::SetCursorPosX(abs_pos.x + 160);
							ImGui::SetCursorPosY(abs_pos.y + 10);
							if (ImGui::Button("Test", { 100, 80 })) {
								/*if (isPreview)
								{
									subRS->stop();//stop preview stream and go through the process to clean the stream.
									std::this_thread::sleep_for(std::chrono::milliseconds(300));
									isPreview = false;
								}*/
								startStreams(baseline_mm, intrin, sfIdx, subDepth, subColor, mViewer_model);
								gCalculate_sts = -1;
								gStartTesting = true;
								gJoinTrigger = true;
								showResult = false;
								isPreview = false;
								/*subRS->stream_enabled[sfIdx.depthSIdx] = true;
								subRS->stream_enabled[sfIdx.infra2SIdx] = true;
								subRS->ui.selected_format_id[sfIdx.infra1SIdx] = sfIdx.infra1Y8FIdx;//change the infrared1 format from RGB to Y8
								auto profiles = subRS->get_selected_profiles();
								auto profilesColor = subColor->get_selected_profiles();
								//get baseline while test
								stream_profile left_s, right_s;
								for (auto p : profiles)
								{
									if (p.stream_index() == 2 && p.stream_type() == RS2_STREAM_INFRARED)
										right_s = p;
									if (p.stream_index() == 0 && p.stream_type() == RS2_STREAM_DEPTH)
										left_s = p;
								}
								auto extrin = (left_s).get_extrinsics_to(right_s);
								baseline_mm = fabs(extrin.translation[0]) * 1000;  // baseline in mm
								intrin = left_s.as<video_stream_profile>().get_intrinsics();

								subRS->play(profiles);
								std::this_thread::sleep_for(std::chrono::milliseconds(10));
								subColor->play(profilesColor);

								for (auto&& profile : profiles)
									mViewer_model.streams[profile.unique_id()].dev = subRS;
								for (auto&& profile : profilesColor)
									mViewer_model.streams[profile.unique_id()].dev = subColor;*/
#ifdef _WIN32
								SetEvent(gStartTestingEvent);
#else
								pthread_mutex_lock(&testMtx);
								testSts |= TEST;
								pthread_cond_signal(&testCon);
								pthread_mutex_unlock(&testMtx);
#endif
							}
							if (subDepth->streaming && !isPreview)//test is running
							{
								if (!gStartTesting)//test complete
								{
									subDepth->stop();
									subColor->stop();
									std::this_thread::sleep_for(std::chrono::milliseconds(300));
									//subDepth->stream_enabled[sfIdx.depthSIdx] = false;
									//subDepth->stream_enabled[sfIdx.infra2SIdx] = false;
									showResult = true;
								}
							}
						}


						isStreaming = subDepth->streaming;
						//1.0.8 backup, should be removed after 1.0.9 testing
						/*if (!isPreview && !gStartTesting)
						{
							ImGui::SetCursorPosX(abs_pos.x + 25);
							ImGui::SetCursorPosY(abs_pos.y + 10);
							if (ImGui::Button("Preview", { 100, 80 }))
							{
								//enable infrared 1 stream with RGB format for preview
								subRS->stream_enabled[sfIdx.infra1SIdx] = true;
								if (sfIdx.infra1ColorFIdx != -1)
									subRS->ui.selected_format_id[sfIdx.infra1SIdx] = sfIdx.infra1ColorFIdx;
								else
									subRS->ui.selected_format_id[sfIdx.infra1SIdx] = sfIdx.infra1Y8FIdx;
								showResult = false;
								isPreview = true;
								auto profiles = subRS->get_selected_profiles();
								subRS->play(profiles);
								for (auto&& profile : profiles)
								{
									mViewer_model.streams[profile.unique_id()].dev = subRS;
								}
							}
						}
						else {
							ImGui::GetWindowDrawList()->AddRectFilled({ abs_pos.x + 25, abs_pos.y + 10 }, { pos.x + 135, abs_pos.y + 90 }, ImColor(from_rgba(114, 109, 117, 0xff)));
							ImGui::SetCursorPosX(abs_pos.x + 35);
							ImGui::SetCursorPosY(abs_pos.y + 40);
							ImGui::TextDisabled("Preview");
						}
						
						if(gStartTesting)
						{
							ImGui::GetWindowDrawList()->AddRectFilled({ abs_pos.x + 160, abs_pos.y + 10 }, { pos.x + 270, abs_pos.y + 90 }, ImColor(from_rgba(114,109, 117, 0xff)));
							ImGui::SetCursorPosX(abs_pos.x + 185);
							ImGui::SetCursorPosY(abs_pos.y + 40);
							ImGui::TextDisabled("Test");
						}
						else
						{
							ImGui::SetCursorPosX(abs_pos.x + 160);
							ImGui::SetCursorPosY(abs_pos.y + 10);
							if (ImGui::Button("Test", { 100, 80 })) {
								if(isPreview)
								{
									subRS->stop();//stop preview stream and go through the process to clean the stream.
									std::this_thread::sleep_for(std::chrono::milliseconds(300));
									isPreview = false;	
								}
								gCalculate_sts = -1;
								gStartTesting = true;
								subRS->stream_enabled[sfIdx.depthSIdx] = true;
								subRS->stream_enabled[sfIdx.infra2SIdx] = true;
								subRS->ui.selected_format_id[sfIdx.infra1SIdx] = sfIdx.infra1Y8FIdx;//change the infrared1 format from RGB to Y8
								auto profiles = subRS->get_selected_profiles();
								auto profilesColor = subColor->get_selected_profiles();
								//get baseline while test
								stream_profile left_s, right_s;
								for (auto p : profiles)
								{
									if (p.stream_index() == 2 && p.stream_type() == RS2_STREAM_INFRARED)
										right_s = p;
									if (p.stream_index() == 0 && p.stream_type() == RS2_STREAM_DEPTH)
										left_s = p;
								}
								auto extrin = (left_s).get_extrinsics_to(right_s);
								baseline_mm = fabs(extrin.translation[0]) * 1000;  // baseline in mm
								intrin = left_s.as<video_stream_profile>().get_intrinsics();

								subRS->play(profiles);
								std::this_thread::sleep_for(std::chrono::milliseconds(10));
								subColor->play(profilesColor);

								for (auto&& profile : profiles)
									mViewer_model.streams[profile.unique_id()].dev = subRS;
								for (auto&& profile : profilesColor)
									mViewer_model.streams[profile.unique_id()].dev = subColor;
#ifdef _WIN32
								SetEvent(gStartTestingEvent);
#else
								pthread_mutex_lock(&testMtx);
								testSts |= TEST;
								pthread_cond_signal(&testCon);
								pthread_mutex_unlock(&testMtx);
#endif
								gJoinTrigger = true;
								showResult = false;
							}
							if (subRS->streaming && !isPreview)//test is running
							{
								if (!gStartTesting)//test complete
								{
									subRS->stop();
									subColor->stop();
									std::this_thread::sleep_for(std::chrono::milliseconds(300));
									subRS->stream_enabled[sfIdx.depthSIdx] = false;
									subRS->stream_enabled[sfIdx.infra2SIdx] = false;
									showResult = true;
								}
							}
						}*/
								
						ImGui::SetCursorPosX(70);
						ImGui::SetCursorPosY(abs_pos.y + 120);

						abs_pos = ImGui::GetCursorScreenPos();

						if (showResult)
						{
							if (gCalculate_sts == -1)
								gCalculate_sts = calTestResult(testResult);
							//if (calTestResult(testResult))
							if(gCalculate_sts == 1)
							{
								ImGui::GetWindowDrawList()->AddRectFilled({ abs_pos.x, abs_pos.y }, { abs_pos.x + 160, abs_pos.y + 150 }, ImColor(from_rgba(0, 255, 0, 0xff)));
								ImGui::SetCursorPosX(abs_pos.x + 55);
								ImGui::SetCursorPosY(abs_pos.y + 65);
								ImGui::Text("Pass");
							}
							else
							{
								ImGui::GetWindowDrawList()->AddRectFilled({ abs_pos.x, abs_pos.y }, { abs_pos.x + 160, abs_pos.y + 150 }, ImColor(from_rgba(255, 0, 0, 0xff)));
								ImGui::SetCursorPosX(abs_pos.x + 55);
								ImGui::SetCursorPosY(abs_pos.y + 65);
								ImGui::Text("Fail");
							}
						}
                    }
                    catch (const rs2::error& e)
                    {
                        error_message = error_to_string(e);
                    }
                    catch (const std::exception& e)
                    {
                        error_message = e.what();
                    }

					ImGui::PopStyleColor(3);
                    ImGui::PopFont();
                }
            }
            
        }
        else
        {
            const ImVec2 pos = ImGui::GetCursorScreenPos();
            ImRect bb(pos, ImVec2(pos.x + ImGui::GetContentRegionAvail().x, pos.y + ImGui::GetContentRegionAvail().y));
            ImGui::GetWindowDrawList()->AddRectFilled(bb.GetTL(), bb.GetBR(), ImColor(dark_window_background));

            mViewer_model.show_no_device_overlay(selected_font, 50, panel_y + 50);
        }

        /*ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();*/

		bool calWPC = false;
		//temporily comment out, waiting for more mature RMS calculation.
		if (isStreaming)
			calWPC = true;
		points p;
		//pointcloud pc;
		texture_buffer* texture_frame = nullptr;
        // Fetch frames from queues
        for (auto&& device_model : device_models)
            for (auto&& sub : device_model.subdevices)
            {
                sub->queues.foreach([&](frame_queue& queue)
                {
                    try
                    {
                        frame f;
						if (queue.poll_for_frame(&f))//filter will be applied here if apply filiter is activated.
						{
							auto texture = mViewer_model.upload_frame(std::move(f), &p);
							//select RGB as texture
							if (f.get_profile().format() == RS2_FORMAT_RGB8)
								texture_frame = texture;
						}
                    }
                    catch (const rs2::error& ex)
                    {
                        error_message = error_to_string(ex);
                        sub->stop();
                    }
                    catch (const std::exception& ex)
                    {
                        error_message = ex.what();
                        sub->stop();
                    }
                });
            }

		/*char temp[50];
		sprintf(temp, "BEFORE GC");
		log(RS2_LOG_SEVERITY_DEBUG, temp);
		mViewer_model.gc_streams();
		sprintf(temp, "AFTER GC");
		log(RS2_LOG_SEVERITY_DEBUG, temp);*/
		
#pragma endregion UI

		rect rect = { panel_width, panel_y, w - panel_width, (float)h - panel_y - output_height };
		
		//auto layout = mViewer_model.calc_layout(rect, calWPC);
		auto layout = mViewer_model.calc_layout(rect, isStreaming);

		//if (gStartTesting && calWPC)
		if (isStreaming)
		{
			bool paused = false;
			mViewer_model.show_3dviewer_header(font_14, layout[DUMMY_PC_STREAM_ID], paused);
			//display point cloud inside the rect
			mViewer_model.update_3d_camera(layout[DUMMY_PC_STREAM_ID], mouse, false);
			
			if (auto _frame = mViewer_model.streams[sfIdx.depthSIdx].texture->get_last_frame())
			{
				ImGui::PushFont(selected_font);
				char temp[50];
				video_frame vframe = _frame.as<video_frame>();
				calPlane((const uint16_t*)vframe.get_data(), vframe.get_width(), vframe.get_height(), 0.001, &intrin, &(mViewer_model.roi_rect), angles);
				ImGui::SetCursorPosX(0); ImGui::Text("Angle : "); getAngleSuggestion(angles, temp);
				ImGui::SameLine(); ImGui::SetCursorPosX(col1+30); ImGui::Text("%s", temp);
				if (gStartTesting)
				{
					ImGui::Text("Testing started");
				}
				ImGui::PopFont();
			}
			mViewer_model.render_3d_view(layout[DUMMY_PC_STREAM_ID], texture_frame, p);
		}
		
		ImGui::End();
		ImGui::PopStyleVar();
		ImGui::PopStyleColor();


		glViewport(0, 0,
			static_cast<int>(ImGui::GetIO().DisplaySize.x),
			static_cast<int>(ImGui::GetIO().DisplaySize.y));
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glfwGetWindowSize(window, &w, &h);
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(0, w, h, 0, -1, +1);

		//if(gStartTesting && calWPC)
		if (isStreaming)
			mViewer_model.display_3d_view(layout[DUMMY_PC_STREAM_ID]);

		for (auto &&kvp : layout)
		{
			auto&& view_rect = kvp.second;
			auto stream = kvp.first;
			auto&& stream_mv = mViewer_model.streams[stream];
			auto&& stream_size = stream_mv.size;
			auto stream_rect = view_rect.adjust_ratio(stream_size);

			if (stream == DUMMY_PC_STREAM_ID)
			{
					
				continue;
			}

			stream_mv.show_frame(stream_rect, mouse, error_message);

			if (stream == sfIdx.depthSIdx)//depth stream
			{
				GLfloat x = stream_rect.x + stream_rect.w * (0.5 - 0.5 * gConfig.ROIPercent * 0.01);
				GLfloat y = stream_rect.y + stream_rect.h * (0.5 - 0.5 * gConfig.ROIPercent * 0.01);
				GLfloat w = (stream_rect.x + stream_rect.w * (0.5 + 0.5 * gConfig.ROIPercent * 0.01)) - x;
				GLfloat h = (stream_rect.y + stream_rect.h * (0.5 + 0.5 * gConfig.ROIPercent * 0.01)) - y;
				glBegin(GL_LINE_STRIP);
				glColor4f(1, 1, 1, 1);

				glVertex2f(x, y);
				glVertex2f(x, y + h);
				glVertex2f(x + w, y + h);
				glVertex2f(x + w, y);
				glVertex2f(x, y);
				glEnd();
			}

			//three area testing, comment out to use percentage of whole area
			/*if (stream == sfIdx.depthSIdx)//depth stream
			{
				GLfloat x = stream_rect.x + (stream_rect.w / 3);
				GLfloat y = stream_rect.y + (stream_rect.h / 3);
				GLfloat w = stream_rect.w / 3;
				GLfloat h = stream_rect.h / 3;
				glBegin(GL_LINE_STRIP);
				glColor4f(1, 1, 1, 1);*/

				/*
				_____w_____|______w_____|_______w_______
				|                       |              |
				|                       |       UR     |
				|        (x,y)__________|______________|
				|          |            |              |
				|          |     C      |              |
				|          |____________|______________|
				|                       |              |
				|                       |       BR     |
				|_______________________|______________|
				*/
				/*GLfloat drawEdgeOffset = stream_rect.w * gOffsetRatio;

				//center rectangle
				glVertex2f(x, y);
				glVertex2f(x, y + h);
				glVertex2f(x + w, y + h);
				glVertex2f(x + w, y);
				glVertex2f(x, y);
				glEnd();

				glBegin(GL_LINE_STRIP);
				//upper right rectangle
				glVertex2f(x + w, stream_rect.y);
				//glVertex2f(x + (2 * w) - drawEdgeOffset, stream_rect.y);
				//glVertex2f(x + (2 * w) - drawEdgeOffset, y);
				glVertex2f(x + (2 * w) - gConfig.edgeOffset, stream_rect.y);
				glVertex2f(x + (2 * w) - gConfig.edgeOffset, y);
				glVertex2f(x + w, y);
				glVertex2f(x + w, stream_rect.y);
				glEnd();

				glBegin(GL_LINE_STRIP);
				//bottom right rectangle
				glVertex2f(x + w, y + h);
				//glVertex2f(x + (2 * w) - drawEdgeOffset, y + h);
				//glVertex2f(x + (2 * w) - drawEdgeOffset, y + (2 * h));
				glVertex2f(x + (2 * w) - gConfig.edgeOffset, y + h);
				glVertex2f(x + (2 * w) - gConfig.edgeOffset, y + (2 * h));
				glVertex2f(x + w, y + (2 * h));
				glVertex2f(x + w, y + h);
				glEnd();
			}*/

			if (gStartCapture)
			{
#ifdef _WIN32
				std::string filename = resultFolder + "\\" + device_models[0].dev.get_info(RS2_CAMERA_INFO_NAME) + "_" + device_models[0].dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) + "_" + stream_mv.dev->stream_display_names[stream] + "_" + std::to_string(testIdx);
#else
				std::string filename = resultFolder + "/" + device_models[0].dev.get_info(RS2_CAMERA_INFO_NAME) + "_" + device_models[0].dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) + "_" + stream_mv.dev->stream_display_names[stream] + "_" + std::to_string(testIdx);
#endif

				auto frame = stream_mv.texture->get_last_frame();
				if (frame)
				{
					colorizer colorRizer;
					frameData[saveImgidx].filename = filename;
					if (stream == sfIdx.depthSIdx)
					{
						int w = frame.as<video_frame>().get_width();
						int h = frame.as<video_frame>().get_height();
						depthImg = new unsigned short[w*h];
						memcpy(depthImg, frame.get_data(), w*h * 2);
						//temporily comment out, waiting for more mature RMS calculation.
						//frameData[saveImgidx]._points = new points(mViewer_model.pc.get_points());
						frameData[saveImgidx]._points = new points(p.get());
						frameData[saveImgidx]._frame = new rs2::frame(colorRizer.colorize(frame));
					}
					else
						frameData[saveImgidx]._frame = new rs2::frame(std::move(frame));
					frameData[saveImgidx].not_model = &mViewer_model.not_model;

#ifdef _WIN32
					saveThread[saveImgidx] = new std::thread(saveFrameThreadFun, frameData[saveImgidx]);
#else
					pthread_create(&saveThread[saveImgidx], NULL, saveFrameThreadFun, (void*)&frameData[saveImgidx]);
#endif
					if (stream == sfIdx.depthSIdx)//depth stream
					{
#ifdef _WIN32
						testResult.csvFileName = resultFolder + "\\" + device_models[0].dev.get_info(RS2_CAMERA_INFO_NAME) + "_" + device_models[0].dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) + "_" + std::to_string(testIdx);
						if (!ends_with(to_lower(testResult.csvFileName), ".csv")) testResult.csvFileName += ".csv";
						calResultThread = new std::thread(calResultThreadFun, frameData[saveImgidx]._frame, depthImg, &testResult, baseline_mm, intrin);
						/*char temp[50];
						sprintf(temp, "calResultThread start");
						log(RS2_LOG_SEVERITY_DEBUG, temp);*/
#else
						testResult.csvFileName = resultFolder + "/" + device_models[0].dev.get_info(RS2_CAMERA_INFO_NAME) + "_" + device_models[0].dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) + "_" + std::to_string(testIdx);
						if (!ends_with(to_lower(testResult.csvFileName), ".csv")) testResult.csvFileName += ".csv";
						testDataSet.frame = frameData[saveImgidx]._frame; testDataSet.data = depthImg; testDataSet.result = &testResult; testDataSet.baseline_mm = baseline_mm; testDataSet.intrin = intrin;
						pthread_create(&calResultThread, NULL, calResultThreadFun, (void*)&testDataSet);
#endif
					}
					saveImgidx++;
				}
			}
		}
		
		//reset the capture flag only when saveImag did happen before, checking saveImgidx
		if (gStartCapture && saveImgidx != 0)
		{
			saveImgidx = 0;
			gStartCapture = false;
			testIdx++;
		}

		//Means testing has ended, join all the thread
		if (!gStartTesting && gJoinTrigger)
		{
			/*char temp[50];
			sprintf(temp, "stop thread");
			log(RS2_LOG_SEVERITY_DEBUG, temp);*/
			gJoinTrigger = false;
#ifdef _WIN32
			ResetEvent(gSaveImageEvent);
			for (int i = 0; i < 4; i++)
			{
				if(saveThread[i] != nullptr && saveThread[i]->joinable())
					saveThread[i]->join();
				frameData[i].clear();
				delete saveThread[i];
			}

			if (calResultThread != nullptr && calResultThread->joinable())
			{
				calResultThread->join();
				delete calResultThread;
			}
#else
			for(int i=0; i<3; i++)
			{
				pthread_join(saveThread[i], NULL);
				frameData[i].clear();
			}
			pthread_join(calResultThread, NULL);
#endif
			//
			delete depthImg;
			/*sprintf(temp, "end stop thread");
			log(RS2_LOG_SEVERITY_DEBUG, temp);*/
		}

		// Metadata overlay windows shall be drawn after textures to preserve z-buffer functionality
		for (auto &&kvp : layout)
		{
			if (mViewer_model.streams[kvp.first].metadata_displayed)
				mViewer_model.streams[kvp.first].show_metadata(mouse);
		}


		mViewer_model.not_model.draw(font_14, w, h);

		mViewer_model.popup_if_error(selected_font, error_message);

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();

        ImGui::Render();
        glfwSwapBuffers(window);
        mouse.mouse_wheel = 0;

        // Yeild the CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

#ifdef _WIN32
	SetEvent(gStopThreadEvent);
	if(testingThread.joinable())
		testingThread.join();
	CloseHandle(gStartTestingEvent);
	CloseHandle(gStopThreadEvent);
	CloseHandle(gSaveFinishEvent);
	CloseHandle(gSaveImageEvent);
#else
	pthread_mutex_lock(&testMtx);
	testSts |= ENDTEST;
	pthread_cond_signal(&testCon);
	pthread_mutex_unlock(&testMtx);
	pthread_join(testThread, NULL);
#endif

    // Stop all subdevices
    for (auto&& device_model : device_models)
        for (auto&& sub : device_model.subdevices)
        {
            if (sub->streaming)
                sub->stop();
        }

    // Cleanup
	//mViewer_model.pc.stop();
    ImGui_ImplGlfw_Shutdown();
    glfwTerminate();


    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}

#ifdef WIN32
int CALLBACK WinMain(
    _In_ HINSTANCE hInstance,
    _In_ HINSTANCE hPrevInstance,
    _In_ LPSTR     lpCmdLine,
    _In_ int       nCmdShow

) {
    main(0, nullptr);
}
#endif
