#include <stdio.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include<cstdlib>

using namespace std;
using namespace cv;

int main(){
	Mat train_img = imread("src.png"), hsv;
	cvtColor(train_img, hsv, CV_BGR2HSV);
	string temp1, temp2;

	int mode = 1;
	if (mode == 0){

	}
	else if (mode == 1){
		// 璸衡DATA羆计

		fstream file("000.txt", ios::in);
		int data_count = 0;
		while (getline(file, temp1))
			data_count++;
		file.close();

		//  trainning mat

		Mat train_data(data_count, 20, CV_32F);//ミ蝴痻皚 820*20篈虫硄笵 float 疊翴计
		Mat train_label(data_count, 1, CV_32F);//ミ蝴痻皚 820*1篈虫硄笵 float 疊翴计
		file.open("000.txt", ios::in);
		for (int c = 0; c < data_count; c++){
			file >> train_label.at<float>(c, 0);
			for (int i = 0; i < 20; i++){
				file >> temp1;
				if (i < 10)
					temp2.assign(temp1, 2, string::npos);
				else
					temp2.assign(temp1, 3, string::npos);
				train_data.at<float>(c, i) = stof(temp2);
			}
		}


		int hist[2][255];
		for (int i = 0; i < 2; i++)
		for (int j = 0; j < 255; j++)
			hist[i][j] = 0;

		for (int i = 0; i < hsv.rows; i++)
		for (int j = 0; j < hsv.cols; j++){
			hist[0][hsv.at<Vec3b>(i, j)[0]]++;
			hist[1][hsv.at<Vec3b>(i, j)[1]]++;
		}

		int max0 = 0;
		int max1 = 0;
		for (int i = 0; i < 255; i++){
			if (hist[0][i] > hist[0][max0])
				max0 = i;
			if (hist[1][i] > hist[1][max1])
				max1 = i;
		}

		TickMeter Time_S;
		Time_S.start();


		// train 癡絤把计
		/*
		CvSVMParams param;
		param.svm_type = CvSVM::C_SVC;
		param.kernel_type = CvSVM::RBF;
		CvSVM svm;
		svm.train_auto(train_data, train_label, Mat(), Mat(), param);
		*/



		//ada癡絤把计
		CvBoostParams param;
		param.boost_type = CvBoost::GENTLE;
		param.weak_count = 30; //畓だ摸竟计
		CvBoost adaboost;
		adaboost.train(train_data, CV_ROW_SAMPLE, train_label, Mat(), Mat(), Mat(), Mat(), param, false);
		//bool CvBoost::train(const CvMat* _train_data, int _tflag,const CvMat* _responses, const CvMat* _var_idx = 0,const CvMat* _sample_idx = 0, const CvMat* _var_type = 0,const CvMat* _missing_mask = 0,CvBoostParams params = CvBoostParams(),bool update = false);
		//split_criteria
		//weight_trim_rate



		// predict
		/*
		Mat out(hsv.rows, hsv.cols, CV_8UC3);
		Mat pix(1, 20, CV_32F);
		float response;
		for (int i = 0; i < train_img.rows; i++)
		for (int j = 0; j < train_img.cols; j++){
			pix.at<float>(0, 0) = max0;
			pix.at<float>(0, 1) = max1;
			for (int ii = -1; ii < 2; ii++)
			for (int jj = -1; jj < 2; jj++){
				pix.at<float>(0, 6 * ii + 2 * jj + 10) = hsv.at<Vec3b>(i, j)[0];
				pix.at<float>(0, 6 * ii + 2 * jj + 11) = hsv.at<Vec3b>(i, j)[1];
			}
			response = svm.predict(pix);
			if (response == 0){
				out.at<Vec3b>(i, j)[0] = train_img.at<Vec3b>(i, j)[0];
				out.at<Vec3b>(i, j)[1] = train_img.at<Vec3b>(i, j)[1];
				out.at<Vec3b>(i, j)[2] = train_img.at<Vec3b>(i, j)[2];
			}
			else if (response == 1){
				out.at<Vec3b>(i, j)[0] = 255;
				out.at<Vec3b>(i, j)[1] = 255;
				out.at<Vec3b>(i, j)[2] = 255;
			}
			else{
				out.at<Vec3b>(i, j)[0] = 0;
				out.at<Vec3b>(i, j)[1] = 0;
				out.at<Vec3b>(i, j)[2] = 0;
			}
		}
		*/

		//adapredict
		Mat out(hsv.rows, hsv.cols, CV_8UC3);
		Mat pix(1, 20, CV_32F);
		float response;
		for (int i = 0; i < train_img.rows; i++)
		for (int j = 0; j < train_img.cols; j++){
			pix.at<float>(0, 0) = max0;
			pix.at<float>(0, 1) = max1;
			for (int ii = -1; ii < 2; ii++)
			for (int jj = -1; jj < 2; jj++){
				pix.at<float>(0, 6 * ii + 2 * jj + 10) = hsv.at<Vec3b>(i, j)[0];
				pix.at<float>(0, 6 * ii + 2 * jj + 11) = hsv.at<Vec3b>(i, j)[1];
			}
			response = adaboost.predict(pix);
			if (response == 0){
				out.at<Vec3b>(i, j)[0] = train_img.at<Vec3b>(i, j)[0];
				out.at<Vec3b>(i, j)[1] = train_img.at<Vec3b>(i, j)[1];
				out.at<Vec3b>(i, j)[2] = train_img.at<Vec3b>(i, j)[2];
			}
			else if (response == 1){
				out.at<Vec3b>(i, j)[0] = 255;
				out.at<Vec3b>(i, j)[1] = 255;
				out.at<Vec3b>(i, j)[2] = 255;
			}
			else{
				out.at<Vec3b>(i, j)[0] = 0;
				out.at<Vec3b>(i, j)[1] = 0;
				out.at<Vec3b>(i, j)[2] = 0;
			}
		}


		Time_S.stop();
		cout << "Predict time      : " << Time_S.getTimeSec() << "(sec)" << endl;

		imwrite("out-adaboost.png", out);
		cvWaitKey(0);

		system("pause");

	}
}
