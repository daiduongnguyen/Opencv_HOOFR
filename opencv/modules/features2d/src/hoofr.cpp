/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/** Authors: Ethan Rublee, Vincent Rabaud, Gary Bradski */

#include "precomp.hpp"
#include "opencl_kernels_features2d.hpp"
#include <stdio.h>
#include <iterator>

#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <bitset>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <string.h>
#include <time.h>
#include <omp.h>

#ifndef CV_IMPL_ADD
#define CV_IMPL_ADD(x)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cv
{

using namespace std;

const float HOOFR_HARRIS_K = 0.04f;
const float HESSIAN_K = 1.0f;

static const double HOOFR_extractor_SQRT2 = 1.4142135623731;
static const double HOOFR_extractor_INV_SQRT2 = 1.0 / HOOFR_extractor_SQRT2;
static const double HOOFR_extractor_LOG2 = 0.693147180559945;
static const int HOOFR_extractor_NB_ORIENTATION = 256;
static const int HOOFR_extractor_NB_POINTS = 49;
static const int HOOFR_extractor_SMALLEST_KP_SIZE = 31; // smallest size of keypoints
static const int HOOFR_extractor_NB_SCALES = HOOFR::NB_SCALES;
static const int HOOFR_extractor_NB_PAIRS = HOOFR::NB_PAIRS;
static const int HOOFR_extractor_NB_ORIENPAIRS = HOOFR::NB_ORIENPAIRS;

// default pairs //version 512 bit
/*
static const int OFFR_extractor_DEF_PAIRS[OFFR_extractor::NB_PAIRS] =
{
1168,1169,1170,1171,1172,1173,1174,1175,1160,1161,1162,1163,1164,1165,
1166,1167,1152,1153,1154,1155,1156,1157,1158,1159,860,902,945,989,1034,1080,1127,1121,
812,813,853,854,895,896,938,939,982,983,1027,1028,1073,1074,1120,1113,804,805,811,844,845,
846,886,887,888,929,930,931,973,974,975,1018,1019,1020,1064,1065,1066,1111,1112,1105,796,797,837,838,879,
880,922,923,966,967,1011,1012,1057,1058,1104,1097,152,170,189,209,230,252,275,269,527,520,552,
553,586,587,621,622,657,658,694,695,732,733,771,772,519,512,513,544,545,546,578,579,580,613,614,
615,649,650,651,686,687,688,724,725,726,763,764,757,511,504,536,537,570,571,605,606,641,642,678,679,716,
717,755,756,44,54,65,77,90,104,119,113,292,293,317,318,343,344,370,371,398,399,427,428,457,458,
488,481,291,284,285,308,309,310,334,335,336,361,362,363,389,390,391,418,419,420,448,449,450,479,
480,473,276,277,301,302,327,328,354,355,382,383,411,412,441,442,472,465,0,2,5,9,14,20,27,
21,135,128,144,145,162,163,181,182,201,202,222,223,244,245,267,268,
28,29,37,38,47,48,58,59,70,71,83,84,97,98,112,105,
127,120,121,136,137,138,154,155,156,173,174,175,193,194,195,214,215,216,236,237,238,259,260,253,
1144,1145,1146,1147,1148,1149,1150,1151,1033,944,1122,
1126,1079,988,901,1075,778,701,628,774,739,664,593,735,463,404,349,459,494,433,376,490,819,
814,852,855,894,897,937,940,981,984,1026,1029,1072,1067,1119,1114,810,806,851,847,885,889,928,932,
972,976,1017,1021,1063,1059,1110,1106,803,798,836,839,878,881,921,924,965,968,1010,1013,1056,1051,
1103,1098,526,521,559,554,585,588,620,623,656,659,693,696,731,734,770,765,518,514,551,
547,577,581,612,616,648,652,685,689,723,719,762,758,510,505,543,538,569,572,604,607,640,
643,677,680,715,718,754,749,299,294,316,319,342,345,369,372,397,400,426,429,456,451,
487,482,290,286,315,311,333,337,360,364,388,392,417,421,447,443,478,474,283,278,300,303,
326,329,353,356,381,384,410,413,440,435,471,466,134,129,151,146,161,164,180,183,200,203,
221,224,243,246,266,261,126,122,143,139,153,157,172,176,192,196,213,217,235,231,258,
254,35,30,36,39,46,49,57,60,69,72,82,85,96,91,111,106,598,671,529,748,13,22,229,270
};
*/

//version 256 bits
static const int HOOFR_extractor_DEF_PAIRS[HOOFR::NB_PAIRS] =
{
1168,1169,1170,1171,1172,1173,1174,1175,1160,1161,1162,1163,1164,1165,
1166,1167,1152,1153,1154,1155,1156,1157,1158,1159,860,902,945,989,1034,1080,1127,1121,
812,813,853,854,895,896,938,939,982,983,1027,1028,1073,1074,1120,1113,804,805,811,844,845,
846,886,887,888,929,930,931,973,974,975,1018,1019,1020,1064,1065,1066,1111,1112,1105,796,797,837,838,879,
880,922,923,966,967,1011,1012,1057,1058,1104,1097,152,170,189,209,230,252,275,269,527,520,552,
553,586,587,621,622,657,658,694,695,732,733,771,772,519,512,513,544,545,546,578,579,580,613,614,
615,649,650,651,686,687,688,724,725,726,763,764,757,511,504,536,537,570,571,605,606,641,642,678,679,716,
717,755,756,44,54,65,77,90,104,119,113,292,293,317,318,343,344,370,371,398,399,427,428,457,458,
488,481,291,284,285,308,309,310,334,335,336,361,362,363,389,390,391,418,419,420,448,449,450,479,
480,473,276,277,301,302,327,328,354,355,382,383,411,412,441,442,472,465,0,2,5,9,14,20,27,
21,135,128,144,145,162,163,181,182,201,202,222,223,244,245,267,268,
28,29,37,38,47,48,58,59,70,71,83,84,97,98,112,105
};



struct PairStat
{
    double mean;
    int idx;
};

struct sortMean
{
    bool operator()( const PairStat& a, const PairStat& b ) const
    {
        return a.mean < b.mean;
    }
};

template<typename _Tp> inline void HOOFR_copyVectorToUMat(const std::vector<_Tp>& v, OutputArray um)
{
    if(v.empty())
        um.release();
    else
        Mat(1, (int)(v.size()*sizeof(v[0])), CV_8U, (void*)&v[0]).copyTo(um);
}

static bool
HOOFR_ocl_HarrisResponses(const UMat& imgbuf,
                    const UMat& layerinfo,
                    const UMat& keypoints,
                    UMat& responses,
                    int nkeypoints, int blockSize, float harris_k)
{
    size_t globalSize[] = {nkeypoints};

    float scale = 1.f/((1 << 2) * blockSize * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;

	ocl::Kernel hr_ker("HOOFR_HarrisResponses", ocl::features2d::hoofr_oclsrc,
                format("-D HOOFR_RESPONSES -D blockSize=%d -D scale_sq_sq=%.12ef -D HOOFR_HARRIS_K=%.12ff", blockSize, scale_sq_sq, harris_k));
    if( hr_ker.empty() )
        return false;

    return hr_ker.args(ocl::KernelArg::ReadOnlyNoSize(imgbuf),
                ocl::KernelArg::PtrReadOnly(layerinfo),
                ocl::KernelArg::PtrReadOnly(keypoints),
                ocl::KernelArg::PtrWriteOnly(responses),
                nkeypoints).run(1, globalSize, 0, true);
}

static bool
HOOFR_ocl_ICAngles(const UMat& imgbuf, const UMat& layerinfo,
             const UMat& keypoints, UMat& responses,
             const UMat& umax, int nkeypoints, int half_k)
{
    size_t globalSize[] = {nkeypoints};

    ocl::Kernel icangle_ker("HOOFR_ICAngle", ocl::features2d::hoofr_oclsrc, "-D HOOFR_ANGLES");
    if( icangle_ker.empty() )
        return false;

    return icangle_ker.args(ocl::KernelArg::ReadOnlyNoSize(imgbuf),
                ocl::KernelArg::PtrReadOnly(layerinfo),
                ocl::KernelArg::PtrReadOnly(keypoints),
                ocl::KernelArg::PtrWriteOnly(responses),
                ocl::KernelArg::PtrReadOnly(umax),
                nkeypoints, half_k).run(1, globalSize, 0, true);
}


static bool
HOOFR_ocl_computeHoofrDescriptors(const UMat& imgbuf, const UMat& layerInfo,
                          const UMat& keypoints, UMat& desc, const UMat& pattern,
                          int nkeypoints, int dsize, int wta_k)
{
    size_t globalSize[] = {nkeypoints};
    
    ocl::Kernel desc_ker("HOOFR_computeDescriptor", ocl::features2d::hoofr_oclsrc,
                         format("-D HOOFR_DESCRIPTORS -D WTA_K=%d", wta_k));
    if( desc_ker.empty() )
        return false;

    return desc_ker.args(ocl::KernelArg::ReadOnlyNoSize(imgbuf),
                         ocl::KernelArg::PtrReadOnly(layerInfo),
                         ocl::KernelArg::PtrReadOnly(keypoints),
                         ocl::KernelArg::PtrWriteOnly(desc),
                         ocl::KernelArg::PtrReadOnly(pattern),
                         nkeypoints, dsize).run(1, globalSize, 0, true);
}


/**
 * Function that computes the Harris responses in a
 * blockSize x blockSize patch at given points in the image
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
static void
HOOFR_HarrisResponses(const Mat& img, const std::vector<Rect>& layerinfo,
                std::vector<KeyPoint>& pts, int blockSize, float harris_k)
{
    CV_Assert( img.type() == CV_8UC1 && blockSize*blockSize <= 2048 );
    size_t ptidx, ptsize = pts.size();

    const uchar* ptr00 = img.ptr<uchar>();
    int step = (int)(img.step/img.elemSize1());
    int r = blockSize/2;

    float scale = 1.f/((1 << 2) * blockSize * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;

    AutoBuffer<int> ofsbuf(blockSize*blockSize);
    int* ofs = ofsbuf;
    for( int i = 0; i < blockSize; i++ )
        for( int j = 0; j < blockSize; j++ )
            ofs[i*blockSize + j] = (int)(i*step + j);

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        int x0 = cvRound(pts[ptidx].pt.x);
        int y0 = cvRound(pts[ptidx].pt.y);
        int z = pts[ptidx].octave;

        const uchar* ptr0 = ptr00 + (y0 - r + layerinfo[z].y)*step + x0 - r + layerinfo[z].x;
        int a = 0, b = 0, c = 0;

        for( int k = 0; k < blockSize*blockSize; k++ )
        {
            const uchar* ptr = ptr0 + ofs[k];
            int Ix = (ptr[1] - ptr[-1])*2 + (ptr[-step+1] - ptr[-step-1]) + (ptr[step+1] - ptr[step-1]);
            int Iy = (ptr[step] - ptr[-step])*2 + (ptr[step-1] - ptr[-step-1]) + (ptr[step+1] - ptr[-step+1]);
            a += Ix*Ix;
            b += Iy*Iy;
            c += Ix*Iy;
        }
        pts[ptidx].response = ((float)a * b - (float)c * c -
                               harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
    }
}
*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static float hessian_dxx[49] = 
{
	0.0008,    0.0015,   -0.0007,   -0.0034,   -0.0007,    0.0015,    0.0008,
    0.0044,    0.0085,   -0.0041,   -0.0191,   -0.0041,    0.0085,    0.0044,
    0.0125,    0.0240,   -0.0117,   -0.0542,   -0.0117,    0.0240,    0.0125,
    0.0177,    0.0340,   -0.0166,   -0.0768,   -0.0166,    0.0340,    0.0177,
    0.0125,    0.0240,   -0.0117,   -0.0542,   -0.0117,    0.0240,    0.0125,
    0.0044,    0.0085,   -0.0041,   -0.0191,   -0.0041,    0.0085,    0.0044,
    0.0008,    0.0015,   -0.0007,   -0.0034,   -0.0007,    0.0015,    0.0008
}; 
static float hessian_dyy[49] = 
{
	0.0008,    0.0044,    0.0125,    0.0177,    0.0125,    0.0044,    0.0008,
    0.0015,    0.0085,    0.0240,    0.0340,    0.0240,    0.0085,    0.0015,
   -0.0007,   -0.0041,   -0.0117,   -0.0166,   -0.0117,   -0.0041,   -0.0007,
   -0.0034,   -0.0191,   -0.0542,   -0.0768,   -0.0542,   -0.0191,   -0.0034,
   -0.0007,   -0.0041,   -0.0117,   -0.0166,   -0.0117,   -0.0041,   -0.0007,
    0.0015,    0.0085,    0.0240,    0.0340,    0.0240,    0.0085,    0.0015,
    0.0008,    0.0044,    0.0125,    0.0177,    0.0125,    0.0044,    0.0008

}; 
static float hessian_dxy[49] = 
{
	0.0009,    0.0035,    0.0050,         0,   -0.0050,   -0.0035,   -0.0009,
    0.0035,    0.0133,    0.0188,         0,   -0.0188,   -0.0133,   -0.0035,
    0.0050,    0.0188,    0.0266,         0,   -0.0266,   -0.0188,   -0.0050,
         0,         0,         0,         0,         0,         0,         0,
   -0.0050,   -0.0188,   -0.0266,         0,    0.0266,    0.0188,    0.0050,
   -0.0035,   -0.0133,   -0.0188,         0,    0.0188,    0.0133,    0.0035,
   -0.0009,   -0.0035,   -0.0050,         0,    0.0050,    0.0035,    0.0009
}; 

static void
HessianResponses(const Mat& img, const std::vector<Rect>& layerinfo,
                 std::vector<KeyPoint>& pts, int blockSize, float hessian_k)
{
	
	CV_Assert( img.type() == CV_8UC1 && blockSize*blockSize <= 2048 );
    size_t ptidx, ptsize = pts.size();
    
    const uchar* ptr00 = img.ptr<uchar>();
    int step = (int)(img.step/img.elemSize1());
    int r = blockSize/2;
    
    AutoBuffer<int> ofsbuf(blockSize*blockSize);    
    int* ofs = ofsbuf;         
    for( int i = 0; i < blockSize; i++ )
        for( int j = 0; j < blockSize; j++ )
            ofs[i*blockSize + j] = (int)(i*step + j);
    
    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        int x0 = cvRound(pts[ptidx].pt.x);
        int y0 = cvRound(pts[ptidx].pt.y);
		int z = pts[ptidx].octave;
		
        const uchar* ptr0 = ptr00 + (y0 - r + layerinfo[z].y)*step + x0 - r + layerinfo[z].x;
        float Dxx = 0.0, Dyy = 0.0, Dxy = 0.0;

        for( int k = 0; k < blockSize*blockSize; k++ )
        {
            const uchar* ptr = ptr0 + ofs[k];          
            Dxx += ((float)ptr[0])*hessian_dxx[k];
            Dyy += ((float)ptr[0])*hessian_dyy[k];
            Dxy += ((float)ptr[0])*hessian_dxy[k];
        }
        pts[ptidx].response = (Dxx * Dyy) - hessian_k*(Dxy * Dxy); 
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
static void HOOFR_ICAngles(const Mat& img, const std::vector<Rect>& layerinfo,
                     std::vector<KeyPoint>& pts, const std::vector<int> & u_max, int half_k)
{
    int step = (int)img.step1();
    size_t ptidx, ptsize = pts.size();

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        const Rect& layer = layerinfo[pts[ptidx].octave];
        const uchar* center = &img.at<uchar>(cvRound(pts[ptidx].pt.y) + layer.y, cvRound(pts[ptidx].pt.x) + layer.x);

        int m_01 = 0, m_10 = 0;

        // Treat the center line differently, v=0
        for (int u = -half_k; u <= half_k; ++u)
            m_10 += u * center[u];

        // Go line by line in the circular patch
        for (int v = 1; v <= half_k; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v*step], val_minus = center[u - v*step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        pts[ptidx].angle = fastAtan2((float)m_01, (float)m_10);
    }
}
*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
static void
computeHoofrDescriptors( const Mat& imagePyramid, const std::vector<Rect>& layerInfo,
                       const std::vector<float>& layerScale, std::vector<KeyPoint>& keypoints,
                       Mat& descriptors, const std::vector<Point>& _pattern, int dsize, int wta_k )
{
    int step = (int)imagePyramid.step;
    int j, i, nkeypoints = (int)keypoints.size();

    for( j = 0; j < nkeypoints; j++ )
    {
        const KeyPoint& kpt = keypoints[j];
        const Rect& layer = layerInfo[kpt.octave];
        float scale = 1.f/layerScale[kpt.octave];
        float angle = kpt.angle;

        angle *= (float)(CV_PI/180.f);
        float a = (float)cos(angle), b = (float)sin(angle);

        const uchar* center = &imagePyramid.at<uchar>(cvRound(kpt.pt.y*scale) + layer.y,
                                                      cvRound(kpt.pt.x*scale) + layer.x);
        float x, y;
        int ix, iy;
        const Point* pattern = &_pattern[0];
        uchar* desc = descriptors.ptr<uchar>(j);

    #if 1
        #define GET_VALUE(idx) \
               (x = pattern[idx].x*a - pattern[idx].y*b, \
                y = pattern[idx].x*b + pattern[idx].y*a, \
                ix = cvRound(x), \
                iy = cvRound(y), \
                *(center + iy*step + ix) )
    #else
        #define GET_VALUE(idx) \
            (x = pattern[idx].x*a - pattern[idx].y*b, \
            y = pattern[idx].x*b + pattern[idx].y*a, \
            ix = cvFloor(x), iy = cvFloor(y), \
            x -= ix, y -= iy, \
            cvRound(center[iy*step + ix]*(1-x)*(1-y) + center[(iy+1)*step + ix]*(1-x)*y + \
                    center[iy*step + ix+1]*x*(1-y) + center[(iy+1)*step + ix+1]*x*y))
    #endif

        if( wta_k == 2 )
        {
            for (i = 0; i < dsize; ++i, pattern += 16)
            {
                int t0, t1, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                val = t0 < t1;
                t0 = GET_VALUE(2); t1 = GET_VALUE(3);
                val |= (t0 < t1) << 1;
                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                val |= (t0 < t1) << 2;
                t0 = GET_VALUE(6); t1 = GET_VALUE(7);
                val |= (t0 < t1) << 3;
                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                val |= (t0 < t1) << 4;
                t0 = GET_VALUE(10); t1 = GET_VALUE(11);
                val |= (t0 < t1) << 5;
                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                val |= (t0 < t1) << 6;
                t0 = GET_VALUE(14); t1 = GET_VALUE(15);
                val |= (t0 < t1) << 7;

                desc[i] = (uchar)val;
            }
        }
        else if( wta_k == 3 )
        {
            for (i = 0; i < dsize; ++i, pattern += 12)
            {
                int t0, t1, t2, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
                val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

                t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

                t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

                t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

                desc[i] = (uchar)val;
            }
        }
        else if( wta_k == 4 )
        {
            for (i = 0; i < dsize; ++i, pattern += 16)
            {
                int t0, t1, t2, t3, u, v, k, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                t2 = GET_VALUE(2); t3 = GET_VALUE(3);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val = k;

                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                t2 = GET_VALUE(6); t3 = GET_VALUE(7);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 2;

                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                t2 = GET_VALUE(10); t3 = GET_VALUE(11);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 4;

                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                t2 = GET_VALUE(14); t3 = GET_VALUE(15);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 6;

                desc[i] = (uchar)val;
            }
        }
        else
            CV_Error( Error::StsBadSize, "Wrong wta_k. It can be only 2, 3 or 4." );
        #undef GET_VALUE
    }
}
*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void initializeHoofrPattern( const Point* pattern0, std::vector<Point>& pattern, int ntuples, int tupleSize, int poolSize )
{
    RNG rng(0x12345678);
    int i, k, k1;
    pattern.resize(ntuples*tupleSize);

    for( i = 0; i < ntuples; i++ )
    {
        for( k = 0; k < tupleSize; k++ )
        {
            for(;;)
            {
                int idx = rng.uniform(0, poolSize);
                Point pt = pattern0[idx];
                for( k1 = 0; k1 < k; k1++ )
                    if( pattern[tupleSize*i + k1] == pt )
                        break;
                if( k1 == k )
                {
                    pattern[tupleSize*i + k] = pt;
                    break;
                }
            }
        }
    }
}

static int HOOFR_bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


static void HOOFR_makeRandomPattern(int patchSize, Point* pattern, int npoints)
{
    RNG rng(0x34985739); // we always start with a fixed seed,
                         // to make patterns the same on each run
    for( int i = 0; i < npoints; i++ )
    {
        pattern[i].x = rng.uniform(-patchSize/2, patchSize/2+1);
        pattern[i].y = rng.uniform(-patchSize/2, patchSize/2+1);
    }
}


static inline float HOOFR_getScale(int level, int firstLevel, double scaleFactor)
{
    return (float)std::pow(scaleFactor, (double)(level - firstLevel));
}


class HOOFR_Impl : public HOOFR
{
public:
    explicit HOOFR_Impl(int _nfeatures, float _scaleFactor, int _nlevels, int _edgeThreshold,
             int _firstLevel, int _WTA_K, int _scoreType, int _patchSize, int _fastThreshold):
        nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
        edgeThreshold(_edgeThreshold), firstLevel(_firstLevel), wta_k(_WTA_K),
        scoreType(_scoreType), patchSize(_patchSize), fastThreshold(_fastThreshold),
        orientationNormalized(true), scaleNormalized(true), 
        patternScale(30.0f), nOctaves(4), extAll(false), nOctaves0(0), selectedPairs0(std::vector<int>())
    {buildPattern();}

    void setMaxFeatures(int maxFeatures) { nfeatures = maxFeatures; }
    int getMaxFeatures() const { return nfeatures; }

    void setScaleFactor(double scaleFactor_) { scaleFactor = scaleFactor_; }
    double getScaleFactor() const { return scaleFactor; }

    void setNLevels(int nlevels_) { nlevels = nlevels_; }
    int getNLevels() const { return nlevels; }

    void setEdgeThreshold(int edgeThreshold_) { edgeThreshold = edgeThreshold_; }
    int getEdgeThreshold() const { return edgeThreshold; }

    void setFirstLevel(int firstLevel_) { firstLevel = firstLevel_; }
    int getFirstLevel() const { return firstLevel; }

    void setWTA_K(int wta_k_) { wta_k = wta_k_; }
    int getWTA_K() const { return wta_k; }

    void setScoreType(int scoreType_) { scoreType = scoreType_; }
    int getScoreType() const { return scoreType; }

    void setPatchSize(int patchSize_) { patchSize = patchSize_; }
    int getPatchSize() const { return patchSize; }

    void setFastThreshold(int fastThreshold_) { fastThreshold = fastThreshold_; }
    int getFastThreshold() const { return fastThreshold; }

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;
    // returns the default norm type
    int defaultNorm() const;

    // Compute the HOOFR_Impl features and descriptors on an image
    void detectAndCompute( InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors, bool useProvidedKeypoints=false );
                     
    vector<int> selectPairs( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints,
                     const double corrThresh = 0.7, bool verbose = true );	
	
protected:

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int edgeThreshold;
    int firstLevel;
    int wta_k;
    int scoreType;
    int patchSize;
    int fastThreshold;
    
    void computeHOOFRdescriptor( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
    void buildPattern();
    uchar meanIntensity( const Mat& image, const Mat& integral, const float kp_x, const float kp_y,
                         const unsigned int scale, const unsigned int rot, const unsigned int point ) const;
	
    bool orientationNormalized; //true if the orientation is normalized, false otherwise
    bool scaleNormalized; //true if the scale is normalized, false otherwise
    double patternScale; //scaling of the pattern
    int nOctaves; //number of octaves
    bool extAll; // true if all pairs need to be extracted for pairs selection

    double patternScale0;
    int nOctaves0;
    vector<int> selectedPairs0;

    struct PatternPoint
    {
        float x; // x coordinate relative to center
        float y; // x coordinate relative to center
        float sigma; // Gaussian smoothing sigma
    };

    struct DescriptionPair
    {
        uchar i; // index of the first point
        uchar j; // index of the second point
    };

    struct OrientationPair
    {
        uchar i; // index of the first point
        uchar j; // index of the second point
        int weight_dx; // dx/(norm_sq))*4096
        int weight_dy; // dy/(norm_sq))*4096
    };

    vector<PatternPoint> patternLookup; // look-up table for the pattern points (position+sigma of all points at all scales and orientation)
    int patternSizes[HOOFR::NB_SCALES]; // size of the pattern at a specific scale (used to check if a point is within image boundaries)
    DescriptionPair descriptionPairs[HOOFR::NB_PAIRS];
    OrientationPair orientationPairs[HOOFR::NB_ORIENPAIRS];  
};

void HOOFR_Impl::buildPattern()
{
    if( patternScale == patternScale0 && nOctaves == nOctaves0 && !patternLookup.empty() )
        return;

    nOctaves0 = nOctaves;
    patternScale0 = patternScale;

    patternLookup.resize(HOOFR_extractor_NB_SCALES*HOOFR_extractor_NB_ORIENTATION*HOOFR_extractor_NB_POINTS);
   
   
    double scaleStep = 1.2;//1.0230518752204628825566;//pow(2.0, (double)(nOctaves - 1)/HOOFR_extractor_NB_SCALES ); // 2 ^ ( (nOctaves-1) /nbScales)
    //double scalingFactor, alpha, beta, theta = 0;
    // pattern definition, radius normalized to 1.0 (outer point position+sigma=1.0)
    const int n[7] = {8,8,8,8,8,8,1}; // number of points on each concentric circle (from outer to inner)    
    const double bigR(2.0/3.0); // bigger radius
    const double smallR(2.0/24.0); // smaller radius
    const double unitSpace( (bigR-smallR)/15.0 ); // define spaces between concentric circles (from center to outer: 1,2,3,4,5,6)
    // radii of the concentric cirles (from outer to inner)
    const double radius[7] = {bigR, bigR-5*unitSpace, bigR-9*unitSpace, bigR-12*unitSpace, bigR-14*unitSpace, smallR, 0.0};
    // sigma of pattern points (each group of 6 points on a concentric cirle has the same sigma)
    const double sigma[7] = {radius[0]/2.0, radius[1]/2.0, radius[2]/2.0,
                             radius[3]/2.0, radius[4]/2.0, radius[5]/2.0,
                             radius[5]/2.0
                            };
    // fill the lookup table
    //#pragma omp parallel for num_threads(4)
    for( int scaleIdx=0; scaleIdx < HOOFR_extractor_NB_SCALES; ++scaleIdx )
    {
        double scalingFactor, alpha, beta, theta = 0;
        patternSizes[scaleIdx] = 0; // proper initialization
        scalingFactor = pow(scaleStep,scaleIdx); //scale of the pattern, scaleStep ^ scaleIdx

        for( int orientationIdx = 0; orientationIdx < HOOFR_extractor_NB_ORIENTATION; ++orientationIdx )
        {
            theta = double(orientationIdx)* 2*CV_PI/double(HOOFR_extractor_NB_ORIENTATION); // orientation of the pattern
            int pointIdx = 0;

            PatternPoint* patternLookupPtr = &patternLookup[0];
            for( size_t i = 0; i < 7; ++i )
            {
                for( int k = 0 ; k < n[i]; ++k )
                {
                    beta = CV_PI/n[i] * (i%2); // orientation offset so that groups of points on each circles are staggered
                    alpha = double(k) * 2*CV_PI/double(n[i])+beta+theta;

                    // add the point to the look-up table
                    PatternPoint& point = patternLookupPtr[ scaleIdx*HOOFR_extractor_NB_ORIENTATION*HOOFR_extractor_NB_POINTS+orientationIdx*HOOFR_extractor_NB_POINTS+pointIdx ];
                    point.x = static_cast<float>(radius[i] * cos(alpha) * scalingFactor * patternScale);
                    point.y = static_cast<float>(radius[i] * sin(alpha) * scalingFactor * patternScale);
                    point.sigma = static_cast<float>(sigma[i] * scalingFactor * patternScale);

                    // adapt the sizeList if necessary
                    const int sizeMax = static_cast<int>(ceil((radius[i]+sigma[i])*scalingFactor*patternScale));
                    if( patternSizes[scaleIdx] < sizeMax )
                        patternSizes[scaleIdx] = sizeMax;

                    ++pointIdx;
                }
            }
        }
    }


    // build the list of orientation pairs
    orientationPairs[0].i=0; orientationPairs[0].j=3; orientationPairs[1].i=0; orientationPairs[1].j=5; orientationPairs[2].i=1; orientationPairs[2].j=4;
    orientationPairs[3].i=1; orientationPairs[3].j=6; orientationPairs[4].i=2; orientationPairs[4].j=5; orientationPairs[5].i=2; orientationPairs[5].j=7;
    orientationPairs[6].i=3; orientationPairs[6].j=6; orientationPairs[7].i=4; orientationPairs[7].j=7; 
    
    orientationPairs[8].i=8; orientationPairs[8].j=11;orientationPairs[9].i=8; orientationPairs[9].j=13; orientationPairs[10].i=9; orientationPairs[10].j=12; 
    orientationPairs[11].i=9; orientationPairs[11].j=14;orientationPairs[12].i=10; orientationPairs[12].j=13; orientationPairs[13].i=10; orientationPairs[13].j=15; 
    orientationPairs[14].i=11; orientationPairs[14].j=14;orientationPairs[15].i=12; orientationPairs[15].j=15; 
    
    orientationPairs[16].i=16; orientationPairs[16].j=19; orientationPairs[17].i=16; orientationPairs[17].j=21;orientationPairs[18].i=17; orientationPairs[18].j=20; 
    orientationPairs[19].i=17; orientationPairs[19].j=22; orientationPairs[20].i=18; orientationPairs[20].j=21;orientationPairs[21].i=18; orientationPairs[21].j=23; 
    orientationPairs[22].i=19; orientationPairs[22].j=22; orientationPairs[23].i=20; orientationPairs[23].j=23;
    
    orientationPairs[24].i=24; orientationPairs[24].j=27; orientationPairs[25].i=24; orientationPairs[25].j=29; orientationPairs[26].i=25; orientationPairs[26].j=28;
    orientationPairs[27].i=25; orientationPairs[27].j=30; orientationPairs[28].i=26; orientationPairs[28].j=29; orientationPairs[29].i=26; orientationPairs[29].j=31;
    orientationPairs[30].i=27; orientationPairs[30].j=30; orientationPairs[31].i=28; orientationPairs[31].j=31; 
    
    
    orientationPairs[32].i=32; orientationPairs[32].j=35;orientationPairs[33].i=32; orientationPairs[33].j=37; orientationPairs[34].i=33; orientationPairs[34].j=36; 
    orientationPairs[35].i=33; orientationPairs[35].j=38;orientationPairs[36].i=34; orientationPairs[36].j=37; orientationPairs[37].i=34; orientationPairs[37].j=39; 
    orientationPairs[38].i=35; orientationPairs[38].j=38;orientationPairs[39].i=36; orientationPairs[39].j=39; 
    
    /*
    orientationPairs[40].i=40; orientationPairs[40].j=43; orientationPairs[41].i=40; orientationPairs[41].j=45;orientationPairs[42].i=41; orientationPairs[42].j=44; 
    orientationPairs[43].i=41; orientationPairs[43].j=46; orientationPairs[44].i=42; orientationPairs[44].j=45;orientationPairs[45].i=42; orientationPairs[45].j=47;
    orientationPairs[46].i=43; orientationPairs[46].j=46;orientationPairs[47].i=44; orientationPairs[47].j=47;
	*/

    //#pragma omp parallel for num_threads(4)
    for(int m = 0; m < HOOFR_extractor_NB_ORIENPAIRS; m++)
    {
        const float dx = patternLookup[orientationPairs[m].i].x-patternLookup[orientationPairs[m].j].x;
        const float dy = patternLookup[orientationPairs[m].i].y-patternLookup[orientationPairs[m].j].y;
        const float norm_sq = (dx*dx+dy*dy);
        orientationPairs[m].weight_dx = int((dx/(norm_sq))*4096.0+0.5);
        orientationPairs[m].weight_dy = int((dy/(norm_sq))*4096.0+0.5);
    }

    // build the list of description pairs
    std::vector<DescriptionPair> allPairs;
    for( unsigned int i = 1; i < (unsigned int)HOOFR_extractor_NB_POINTS; ++i )
    {
        // (generate all the pairs)
        for( unsigned int j = 0; (unsigned int)j < i; ++j )
        {
            DescriptionPair pair = {(uchar)i,(uchar)j};
            allPairs.push_back(pair);
        }
    }
    // Input vector provided
    if( !selectedPairs0.empty() )
    {
        if( (int)selectedPairs0.size() == HOOFR_extractor_NB_PAIRS )
        {
            for( int i = 0; i < HOOFR_extractor_NB_PAIRS; ++i )
                 descriptionPairs[i] = allPairs[selectedPairs0.at(i)];
        }
        else
        {
            printf("ERROR !!! Input vector does not match the required size \n");
            return;
        }
    }
    else // default selected pairs
    {
        //#pragma omp parallel for num_threads(4)
        for( int i = 0; i < HOOFR_extractor_NB_PAIRS; ++i )
             descriptionPairs[i] = allPairs[HOOFR_extractor_DEF_PAIRS[i]];
    }
}


void HOOFR_Impl::computeHOOFRdescriptor( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors ) const
{
    if( image.empty() )
        return;
    if( keypoints.empty() )
        return;
    
    ((HOOFR_Impl*)this)->buildPattern();
	
	
    Mat imgIntegral;
    integral(image, imgIntegral);
           
    std::vector<int> kpScaleIdx(keypoints.size()); // used to save pattern scale index corresponding to each keypoints
    const std::vector<int>::iterator ScaleIdxBegin = kpScaleIdx.begin(); // used in std::vector erase function
    const std::vector<cv::KeyPoint>::iterator kpBegin = keypoints.begin(); // used in std::vector erase function
    const float sizeCst = 1.0;//8.0;//static_cast<float>(HOOFR_extractor_NB_SCALES/(HOOFR_extractor_LOG2* nOctaves));
    //uchar pointsValue[HOOFR_extractor_NB_POINTS];
    //int thetaIdx;
    //int direction0;
    //int direction1;

    // compute the scale index corresponding to the keypoint size and remove keypoints close to the border
    if( scaleNormalized )
    {
        for( size_t k = keypoints.size(); k--; )
        {
            //Is k non-zero? If so, decrement it and continue"
            //kpScaleIdx[k] = max( (int)((keypoints[k].size/HOOFR_extractor_SMALLEST_KP_SIZE - 1.0)*sizeCst+0.5) ,0);
            kpScaleIdx[k] = max( (int)(keypoints[k].octave*sizeCst) ,0);
            if( kpScaleIdx[k] >= HOOFR_extractor_NB_SCALES )
                kpScaleIdx[k] = HOOFR_extractor_NB_SCALES-1;

            if( keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] || //check if the description at this specific position and scale fits inside the image
                 keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
                 keypoints[k].pt.x >= image.cols-patternSizes[kpScaleIdx[k]] ||
                 keypoints[k].pt.y >= image.rows-patternSizes[kpScaleIdx[k]]
               )
            {
                keypoints.erase(kpBegin+k);
                kpScaleIdx.erase(ScaleIdxBegin+k);
            }
        }
    }
    else
    {
        const int scIdx = max( (int)(1.0986122886681*sizeCst+0.5) ,0);
        for( size_t k = keypoints.size(); k--; )
        {
            kpScaleIdx[k] = scIdx; // equivalent to the formule when the scale is normalized with a constant size of keypoints[k].size=3*SMALLEST_KP_SIZE
            if( kpScaleIdx[k] >= HOOFR_extractor_NB_SCALES )
            {
                kpScaleIdx[k] = HOOFR_extractor_NB_SCALES-1;
            }
            if( keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] ||
                keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
                keypoints[k].pt.x >= image.cols-patternSizes[kpScaleIdx[k]] ||
                keypoints[k].pt.y >= image.rows-patternSizes[kpScaleIdx[k]]
               )
            {
                keypoints.erase(kpBegin+k);
                kpScaleIdx.erase(ScaleIdxBegin+k);
            }
        }
    }
	
    // allocate descriptor memory, estimate orientations, extract descriptors

    if( !extAll )
    {
        // extract the best comparisons only
        descriptors = cv::Mat::zeros((int)keypoints.size(), HOOFR_extractor_NB_PAIRS/8, CV_8U);
#if CV_SSE2
		//printf("AVEC CV_SSE2 \n");
        __m128i* ptr= (__m128i*) (descriptors.data + 0*descriptors.step[0]);    
#else
        std::bitset<HOOFR_extractor_NB_PAIRS>* ptr = (std::bitset<HOOFR_extractor_NB_PAIRS>*) (descriptors.data+ 0*descriptors.step[0]);
#endif
        //for( size_t k = keypoints.size(); k--; )
        #pragma omp parallel for num_threads(4)
        for( int k = 0; k < (int)keypoints.size(); k++) 
        {
            uchar pointsValue[HOOFR_extractor_NB_POINTS];			
			int direction0;
			int direction1;
			                
            // get the points intensity value in the un-rotated pattern
            /**********************************/               
            for( int i = 0; i < HOOFR_extractor_NB_POINTS-9; i++)
            {
                pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x,keypoints[k].pt.y, kpScaleIdx[k], 0, i);
            }
            direction0 = 0;
            direction1 = 0;
            for( int m = 0; m < 40; m++)
            {
                //iterate through the orientation pairs
                const int delta = (pointsValue[ orientationPairs[m].i ]-pointsValue[ orientationPairs[m].j ]);
                direction0 += delta*(orientationPairs[m].weight_dx)/2048;
                direction1 += delta*(orientationPairs[m].weight_dy)/2048;
            }				
            keypoints[k].angle = static_cast<float>(atan2((float)direction1,(float)direction0)*(180.0/CV_PI));//estimate orientation
            
        }
    
        #pragma omp parallel for num_threads(4) 
        for( int k = 0; k < (int)keypoints.size(); k++) 
        {   
            uchar pointsValue[HOOFR_extractor_NB_POINTS];
			int thetaIdx = 0;
			
			// estimate orientation (gradient)
            if( !orientationNormalized )
            {
                thetaIdx = 0; // assign 0° to all keypoints
                //keypoints[k].angle = 0.0;
            }
            else
            {
                /**********************************/
                thetaIdx = int(HOOFR_extractor_NB_ORIENTATION*keypoints[k].angle*(1/360.0)+0.5);
                if( thetaIdx < 0 )
                    thetaIdx += HOOFR_extractor_NB_ORIENTATION;

                if( thetaIdx >= HOOFR_extractor_NB_ORIENTATION )
                    thetaIdx -= HOOFR_extractor_NB_ORIENTATION;
            }
                
            // extract descriptor at the computed orientation                      
            for( int i = 0; i < HOOFR_extractor_NB_POINTS; i++)
            {
                pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x,keypoints[k].pt.y, kpScaleIdx[k], thetaIdx, i);
            }
#if CV_SSE2

            // note that comparisons order is modified in each block (but first 128 comparisons remain globally the same-->does not affect the 128,384 bits segmanted matching strategy)
            int cnt = 0;                     
            for( int n = 0;n < HOOFR_extractor_NB_PAIRS/128; n++)
            {
                __m128i result128 = _mm_setzero_si128();
                for( int m = 128/16; m--; cnt += 16 )
                {
                    __m128i operand1 = _mm_set_epi8(
                        pointsValue[descriptionPairs[cnt+0].i],
                        pointsValue[descriptionPairs[cnt+1].i],
                        pointsValue[descriptionPairs[cnt+2].i],
                        pointsValue[descriptionPairs[cnt+3].i],
                        pointsValue[descriptionPairs[cnt+4].i],
                        pointsValue[descriptionPairs[cnt+5].i],
                        pointsValue[descriptionPairs[cnt+6].i],
                        pointsValue[descriptionPairs[cnt+7].i],
                        pointsValue[descriptionPairs[cnt+8].i],
                        pointsValue[descriptionPairs[cnt+9].i],
                        pointsValue[descriptionPairs[cnt+10].i],
                        pointsValue[descriptionPairs[cnt+11].i],
                        pointsValue[descriptionPairs[cnt+12].i],
                        pointsValue[descriptionPairs[cnt+13].i],
                        pointsValue[descriptionPairs[cnt+14].i],
                        pointsValue[descriptionPairs[cnt+15].i]);

                    __m128i operand2 = _mm_set_epi8(
                        pointsValue[descriptionPairs[cnt+0].j],
                        pointsValue[descriptionPairs[cnt+1].j],
                        pointsValue[descriptionPairs[cnt+2].j],
                        pointsValue[descriptionPairs[cnt+3].j],
                        pointsValue[descriptionPairs[cnt+4].j],
                        pointsValue[descriptionPairs[cnt+5].j],
                        pointsValue[descriptionPairs[cnt+6].j],
                        pointsValue[descriptionPairs[cnt+7].j],
                        pointsValue[descriptionPairs[cnt+8].j],
                        pointsValue[descriptionPairs[cnt+9].j],
                        pointsValue[descriptionPairs[cnt+10].j],
                        pointsValue[descriptionPairs[cnt+11].j],
                        pointsValue[descriptionPairs[cnt+12].j],
                        pointsValue[descriptionPairs[cnt+13].j],
                        pointsValue[descriptionPairs[cnt+14].j],
                        pointsValue[descriptionPairs[cnt+15].j]);

                    __m128i workReg = _mm_min_epu8(operand1, operand2); // emulated "not less than" for 8-bit UNSIGNED integers
                    workReg = _mm_cmpeq_epi8(workReg, operand2);        // emulated "not less than" for 8-bit UNSIGNED integers

                    workReg = _mm_and_si128(_mm_set1_epi16(short(0x8080 >> m)), workReg); // merge the last 16 bits with the 128bits std::vector until full
                    result128 = _mm_or_si128(result128, workReg);
                }
                __m128i* ptr_kn = ptr + k*(HOOFR_extractor_NB_PAIRS/128) + n;               
                (*ptr_kn) = result128;
                //++ptr;
            }
            //ptr -= 8;

#else

            // extracting descriptor preserving the order of SSE version
            std::bitset<HOOFR_extractor_NB_PAIRS>* ptr_k = ptr + k;
            int cnt = 0;
            for( int n = 7; n < HOOFR_extractor_NB_PAIRS; n += 128)
            {
                for( int m = 8; m--; )
                {
                    int nm = n-m;
                    for(int kk = nm+15*8; kk >= nm; kk-=8, ++cnt)
                    {
                        ptr_k->set(kk, pointsValue[descriptionPairs[cnt].i] >= pointsValue[descriptionPairs[cnt].j]);
                    }
                }
            }
            //ptr++;
#endif
        }
    }
    else // extract all possible comparisons for selection
    {
        descriptors = cv::Mat::zeros((int)keypoints.size(), 160, CV_8U);
        std::bitset<1280>* ptr = (std::bitset<1280>*) (descriptors.data+(keypoints.size()-1)*descriptors.step[0]);

        for( size_t k = keypoints.size(); k--; )
        {
            uchar pointsValue[HOOFR_extractor_NB_POINTS];
			int thetaIdx = 0;
			int direction0;
			int direction1;
			
            //estimate orientation (gradient)
            if( !orientationNormalized )
            {
                thetaIdx = 0;//assign 0° to all keypoints
                keypoints[k].angle = 0.0;
            }
            else
            {
                //get the points intensity value in the un-rotated pattern
                /**********************************/
                
                for( int i = HOOFR_extractor_NB_POINTS;i--; )
                    pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x,keypoints[k].pt.y, kpScaleIdx[k], 0, i);

                direction0 = 0;
                direction1 = 0;
                for( int m = 40; m--; )
                {
                    //iterate through the orientation pairs
                    const int delta = (pointsValue[ orientationPairs[m].i ]-pointsValue[ orientationPairs[m].j ]);
                    direction0 += delta*(orientationPairs[m].weight_dx)/2048;
                    direction1 += delta*(orientationPairs[m].weight_dy)/2048;
                }

                keypoints[k].angle = static_cast<float>(atan2((float)direction1,(float)direction0)*(180.0/CV_PI)); //estimate orientation
                
                /**********************************/
                thetaIdx = int(HOOFR_extractor_NB_ORIENTATION*keypoints[k].angle*(1/360.0)+0.5);

                if( thetaIdx < 0 )
                    thetaIdx += HOOFR_extractor_NB_ORIENTATION;

                if( thetaIdx >= HOOFR_extractor_NB_ORIENTATION )
                    thetaIdx -= HOOFR_extractor_NB_ORIENTATION;
            }
            // get the points intensity value in the rotated pattern
            for( int i = HOOFR_extractor_NB_POINTS; i--; )
            {
                pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x,
                                             keypoints[k].pt.y, kpScaleIdx[k], thetaIdx, i);
            }

            int cnt(0);
            for( int i = 1; i < HOOFR_extractor_NB_POINTS; ++i )
            {
                //(generate all the pairs)
                for( int j = 0; j < i; ++j )
                {
                    ptr->set(cnt, pointsValue[i] >= pointsValue[j] );
                    ++cnt;
                }
            }
            --ptr;
        }
    }
}



uchar HOOFR_Impl::meanIntensity( const cv::Mat& image, const cv::Mat& integral,
                            const float kp_x,
                            const float kp_y,
                            const unsigned int scale,
                            const unsigned int rot,
                            const unsigned int point) const
{
       
    // get point position in image
    const PatternPoint& FreakPoint = patternLookup[scale*HOOFR_extractor_NB_ORIENTATION*HOOFR_extractor_NB_POINTS + rot*HOOFR_extractor_NB_POINTS + point];
    const float xf = FreakPoint.x+kp_x;
    const float yf = FreakPoint.y+kp_y;
    const int& imagecols = image.cols;
    // get the sigma:
    const float radius = FreakPoint.sigma;

    // calculate output:
    
    if( radius < 0.5 )
    {
		printf("voila \n");
		const int x = int(xf);
        const int y = int(yf);
        // interpolation multipliers:
        const int r_x = static_cast<int>((xf-x)*1024);
        const int r_y = static_cast<int>((yf-y)*1024);
        const int r_x_1 = (1024-r_x);
        const int r_y_1 = (1024-r_y);
        uchar* ptr = image.data+x+y*imagecols;
        unsigned int ret_val;
        // linear interpolation:
        ret_val = (r_x_1*r_y_1*int(*ptr));
        ptr++;
        ret_val += (r_x*r_y_1*int(*ptr));
        ptr += imagecols;
        ret_val += (r_x*r_y*int(*ptr));
        ptr--;
        ret_val += (r_x_1*r_y*int(*ptr));
        //return the rounded mean
        ret_val += 2 * 1024 * 1024;
        return static_cast<uchar>(ret_val / (4 * 1024 * 1024));       
    }
	
    // expected case:

    // calculate borders
    const int x_left = int(xf-radius+0.5);
    const int y_top = int(yf-radius+0.5);
    const int x_right = int(xf+radius+1.5);//integral image is 1px wider
    const int y_bottom = int(yf+radius+1.5);//integral image is 1px higher
    int ret_val;
    
    
    //if (rot == 0)
    //{
	//	return ( image.at<uchar>((int)(kp_y + patternLookup[scale*HOOFR_extractor_NB_ORIENTATION*HOOFR_extractor_NB_POINTS + point].y),(int)(kp_x + patternLookup[scale*HOOFR_extractor_NB_ORIENTATION*HOOFR_extractor_NB_POINTS + point].x)) );
    //}
	
    ret_val = integral.at<int>(y_bottom,x_right)-integral.at<int>(y_bottom,x_left);//bottom right corner
    ret_val += integral.at<int>(y_top,x_left);
    ret_val -= integral.at<int>(y_top,x_right);
    ret_val = ret_val/( (x_right-x_left)* (y_bottom-y_top) );
    //~ std::cout<<integral.step[1]<<std::endl;

    return static_cast<uchar>(ret_val);
}


/**********SelectPairs function is not completed. It could be rewrite for HOOFR***************************/
vector<int> HOOFR_Impl::selectPairs(const std::vector<Mat>& images
                                        , std::vector<std::vector<KeyPoint> >& keypoints
                                        , const double corrTresh
                                        , bool verbose )
{
    extAll = true;
    // compute descriptors with all pairs
    Mat descriptors;

    if( verbose )
        std::cout << "Number of images: " << images.size() << std::endl;

    for( size_t i = 0;i < images.size(); ++i )
    {
        Mat descriptorsTmp;
        computeHOOFRdescriptor(images[i],keypoints[i],descriptorsTmp);
        descriptors.push_back(descriptorsTmp);
    }

    if( verbose )
        std::cout << "number of keypoints: " << descriptors.rows << std::endl;

    //descriptor in floating point format (each bit is a float)
    Mat descriptorsFloat = Mat::zeros(descriptors.rows, 1176, CV_32F);

    std::bitset<1024>* ptr = (std::bitset<1024>*) (descriptors.data+(descriptors.rows-1)*descriptors.step[0]);
    for( int m = descriptors.rows; m--; )
    {
        for( int n = 1176; n--; )
        {
            if( ptr->test(n) == true )
                descriptorsFloat.at<float>(m,n)=1.0f;
        }
        --ptr;
    }

    std::vector<PairStat> pairStat;
    for( int n = 1176; n--; )
    {
        // the higher the variance, the better --> mean = 0.5
        PairStat tmp = { fabs( mean(descriptorsFloat.col(n))[0]-0.5 ) ,n};
        pairStat.push_back(tmp);
    }

    std::sort( pairStat.begin(),pairStat.end(), sortMean() );

    std::vector<PairStat> bestPairs;
    for( int m = 0; m < 1176; ++m )
    {
        if( verbose )
            std::cout << m << ":" << bestPairs.size() << " " << std::flush;
        double corrMax(0);

        for( size_t n = 0; n < bestPairs.size(); ++n )
        {
            //int idxA = bestPairs[n].idx;
            //int idxB = pairStat[m].idx;
            double corr(0);
            // compute correlation between 2 pairs
            /*corr = fabs(compareHist(descriptorsFloat.col(idxA), descriptorsFloat.col(idxB), CV_COMP_CORREL));*/

            if( corr > corrMax )
            {
                corrMax = corr;
                if( corrMax >= corrTresh )
                    break;
            }
        }

        if( corrMax < corrTresh/*0.7*/ )
            bestPairs.push_back(pairStat[m]);

        if( bestPairs.size() >= 256 )
        {
            if( verbose )
                std::cout << m << std::endl;
            break;
        }
    }

    std::vector<int> idxBestPairs;
    if( (int)bestPairs.size() >= HOOFR_extractor_NB_PAIRS )
    {
        for( int i = 0; i < HOOFR_extractor_NB_PAIRS; ++i )
            idxBestPairs.push_back(bestPairs[i].idx);
    }
    else
    {
        if( verbose )
            std::cout << "correlation threshold too small (restrictive)" << std::endl;          
        //CV_Error(CV_StsError, "correlation threshold too small (restrictive)");
    }
    extAll = false;
    return idxBestPairs;
}



int HOOFR_Impl::descriptorSize() const
{
    return HOOFR_extractor_NB_PAIRS / 8;
}

int HOOFR_Impl::descriptorType() const
{
    return CV_8U;
}

int HOOFR_Impl::defaultNorm() const
{
    return NORM_HAMMING;
}

static void uploadHOOFRKeypoints(const std::vector<KeyPoint>& src, std::vector<Vec3i>& buf, OutputArray dst)
{
    size_t i, n = src.size();
    buf.resize(std::max(buf.size(), n));
    for( i = 0; i < n; i++ )
        buf[i] = Vec3i(cvRound(src[i].pt.x), cvRound(src[i].pt.y), src[i].octave);
    HOOFR_copyVectorToUMat(buf, dst);
}

typedef union if32_t
{
    int i;
    float f;
}
if32_t;

static void uploadHOOFRKeypoints(const std::vector<KeyPoint>& src,
                               const std::vector<float>& layerScale,
                               std::vector<Vec4i>& buf, OutputArray dst)
{
    size_t i, n = src.size();
    buf.resize(std::max(buf.size(), n));
    for( i = 0; i < n; i++ )
    {
        int z = src[i].octave;
        float scale = 1.f/layerScale[z];
        if32_t angle;
        angle.f = src[i].angle;
        buf[i] = Vec4i(cvRound(src[i].pt.x*scale), cvRound(src[i].pt.y*scale), z, angle.i);
    }
    HOOFR_copyVectorToUMat(buf, dst);
}


/** Compute the HOOFR_Impl keypoints on an image
 * @param image_pyramid the image pyramid to compute the features and descriptors on
 * @param mask_pyramid the masks to apply at every level
 * @param keypoints the resulting keypoints, clustered per level
 */
static void HOOFR_computeKeyPoints(const Mat& imagePyramid,
                             const UMat& uimagePyramid,
                             const Mat& maskPyramid,
                             const std::vector<Rect>& layerInfo,
                             const UMat& ulayerInfo,
                             const std::vector<float>& layerScale,
                             std::vector<KeyPoint>& allKeypoints,
                             int nfeatures, double scaleFactor,
                             int edgeThreshold, int patchSize, int scoreType,
                             bool useOCL, int fastThreshold  )
{
    int i, nkeypoints, level, nlevels = (int)layerInfo.size();
    std::vector<int> nfeaturesPerLevel(nlevels);

    // fill the extractors and descriptors for the corresponding scales
    float factor = (float)(1.0 / scaleFactor);
    float ndesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)std::pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( level = 0; level < nlevels-1; level++ )
    {
        nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevel[level];
        ndesiredFeaturesPerScale *= factor;
    }
    nfeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    // Make sure we forget about what is too close to the boundary
    //edge_threshold_ = std::max(edge_threshold_, patch_size_/2 + kKernelWidth / 2 + 2);

    // pre-compute the end of a row in a circular patch
    int halfPatchSize = patchSize / 2;
    std::vector<int> umax(halfPatchSize + 2);

    int v, v0, vmax = cvFloor(halfPatchSize * std::sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * std::sqrt(2.f) / 2);
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(std::sqrt((double)halfPatchSize * halfPatchSize - v * v));

    // Make sure we are symmetric
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    allKeypoints.clear();
    std::vector<KeyPoint> keypoints;
    std::vector<int> counters(nlevels);
    keypoints.reserve(nfeaturesPerLevel[0]*2);

    for( level = 0; level < nlevels; level++ )
    {
        int featuresNum = nfeaturesPerLevel[level];
        Mat img = imagePyramid(layerInfo[level]);
        Mat mask = maskPyramid.empty() ? Mat() : maskPyramid(layerInfo[level]);

        // Detect FAST features, 20 is a good threshold
        {
        Ptr<FastFeatureDetector> fd = FastFeatureDetector::create(fastThreshold, true);
        fd->detect(img, keypoints, mask);
        }

        // Remove keypoints very close to the border
        KeyPointsFilter::runByImageBorder(keypoints, img.size(), edgeThreshold);

        // Keep more points than necessary as FAST does not give amazing corners
        KeyPointsFilter::retainBest(keypoints, scoreType == HOOFR_Impl::HESSIAN_SCORE ? 2 * featuresNum : featuresNum);

        nkeypoints = (int)keypoints.size();
        counters[level] = nkeypoints;

        float sf = layerScale[level];
        for( i = 0; i < nkeypoints; i++ )
        {
            keypoints[i].octave = level;
            keypoints[i].size = patchSize*sf;
        }

        std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(allKeypoints));
    }

    std::vector<Vec3i> ukeypoints_buf;

    nkeypoints = (int)allKeypoints.size();
    if(nkeypoints == 0)
    {
        return;
    }
    Mat responses;
    UMat ukeypoints, uresponses(1, nkeypoints, CV_32F);

    // Select best features using the Harris cornerness (better scoring than FAST)
    if( scoreType == HOOFR_Impl::HESSIAN_SCORE )
    {
//       if( useOCL )
//       {
//            uploadHOOFRKeypoints(allKeypoints, ukeypoints_buf, ukeypoints);
//            useOCL = HOOFR_ocl_HarrisResponses( uimagePyramid, ulayerInfo, ukeypoints,
//                                          uresponses, nkeypoints, 7, HOOFR_HARRIS_K );
//            if( useOCL )
//            {
//               CV_IMPL_ADD(CV_IMPL_OCL);
//               uresponses.copyTo(responses);
//                for( i = 0; i < nkeypoints; i++ )
//                    allKeypoints[i].response = responses.at<float>(i);
//            }
//        }

//        if( !useOCL )
//        {
            //HOOFR_HarrisResponses(imagePyramid, layerInfo, allKeypoints, 7, HOOFR_HARRIS_K);
			/****/
			HessianResponses(imagePyramid, layerInfo, allKeypoints, 7, HESSIAN_K);
			/****/
//		  }
        std::vector<KeyPoint> newAllKeypoints;
        newAllKeypoints.reserve(nfeaturesPerLevel[0]*nlevels);

        int offset = 0;
        for( level = 0; level < nlevels; level++ )
        {
            int featuresNum = nfeaturesPerLevel[level];
            nkeypoints = counters[level];
            keypoints.resize(nkeypoints);
            std::copy(allKeypoints.begin() + offset,
                      allKeypoints.begin() + offset + nkeypoints,
                      keypoints.begin());
            offset += nkeypoints;

            //cull to the final desired level, using the new Harris scores.
            KeyPointsFilter::retainBest(keypoints, featuresNum);

            std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(newAllKeypoints));
        }
        std::swap(allKeypoints, newAllKeypoints);
    }

    nkeypoints = (int)allKeypoints.size();
//    if( useOCL )
//    {
//        UMat uumax;
//        if( useOCL )
//            HOOFR_copyVectorToUMat(umax, uumax);

//        uploadHOOFRKeypoints(allKeypoints, ukeypoints_buf, ukeypoints);
//        useOCL = HOOFR_ocl_ICAngles(uimagePyramid, ulayerInfo, ukeypoints, uresponses, uumax,
//                              nkeypoints, halfPatchSize);

//        if( useOCL )
//        {
//            CV_IMPL_ADD(CV_IMPL_OCL);
//            uresponses.copyTo(responses);
//            for( i = 0; i < nkeypoints; i++ )
//                allKeypoints[i].angle = responses.at<float>(i);
//        }
//    }

//    if( !useOCL )
//    {
        //HOOFR_ICAngles(imagePyramid, layerInfo, allKeypoints, umax, halfPatchSize);
//    }

    for( i = 0; i < nkeypoints; i++ )
    {
        float scale = layerScale[allKeypoints[i].octave];
        allKeypoints[i].pt *= scale;
    }
}


/** Compute the HOOFR_Impl features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 * @param descriptors the resulting descriptors
 * @param do_keypoints if true, the keypoints are computed, otherwise used as an input
 * @param do_descriptors if true, also computes the descriptors
 */
void HOOFR_Impl::detectAndCompute( InputArray _image, InputArray _mask,
                                 std::vector<KeyPoint>& keypoints,
                                 OutputArray _descriptors, bool useProvidedKeypoints )
{
    CV_Assert(patchSize >= 2);

    bool do_keypoints = !useProvidedKeypoints;
    bool do_descriptors = _descriptors.needed();

    if( (!do_keypoints && !do_descriptors) || _image.empty() )
        return;

    //ROI handling
    const int HARRIS_BLOCK_SIZE = 9;
    int halfPatchSize = patchSize / 2;
    int border = std::max(edgeThreshold, std::max(halfPatchSize, HARRIS_BLOCK_SIZE/2))+1;

    bool useOCL = ocl::useOpenCL();

    Mat image = _image.getMat(), mask = _mask.getMat();
    if( image.type() != CV_8UC1 )
        cvtColor(_image, image, COLOR_BGR2GRAY);

    int i, level, nLevels = this->nlevels, nkeypoints = (int)keypoints.size();
    bool sortedByLevel = true;

    if( !do_keypoints )
    {
        // if we have pre-computed keypoints, they may use more levels than it is set in parameters
        // !!!TODO!!! implement more correct method, independent from the used keypoint detector.
        // Namely, the detector should provide correct size of each keypoint. Based on the keypoint size
        // and the algorithm used (i.e. BRIEF, running on 31x31 patches) we should compute the approximate
        // scale-factor that we need to apply. Then we should cluster all the computed scale-factors and
        // for each cluster compute the corresponding image.
        //
        // In short, ultimately the descriptor should
        // ignore octave parameter and deal only with the keypoint size.
        nLevels = 0;
        for( i = 0; i < nkeypoints; i++ )
        {
            level = keypoints[i].octave;
            CV_Assert(level >= 0);
            if( i > 0 && level < keypoints[i-1].octave )
                sortedByLevel = false;
            nLevels = std::max(nLevels, level);
        }
        nLevels++;
    }

    std::vector<Rect> layerInfo(nLevels);
    std::vector<int> layerOfs(nLevels);
    std::vector<float> layerScale(nLevels);
    Mat imagePyramid, maskPyramid;
    UMat uimagePyramid, ulayerInfo;

    int level_dy = image.rows + border*2;
    Point level_ofs(0,0);
    Size bufSize((image.cols + border*2 + 15) & -16, 0);

    for( level = 0; level < nLevels; level++ )
    {
        float scale = HOOFR_getScale(level, firstLevel, scaleFactor);
        layerScale[level] = scale;
        Size sz(cvRound(image.cols/scale), cvRound(image.rows/scale));
        Size wholeSize(sz.width + border*2, sz.height + border*2);
        if( level_ofs.x + wholeSize.width > bufSize.width )
        {
            level_ofs = Point(0, level_ofs.y + level_dy);
            level_dy = wholeSize.height;
        }

        Rect linfo(level_ofs.x + border, level_ofs.y + border, sz.width, sz.height);
        layerInfo[level] = linfo;
        layerOfs[level] = linfo.y*bufSize.width + linfo.x;
        level_ofs.x += wholeSize.width;
    }
    bufSize.height = level_ofs.y + level_dy;

    imagePyramid.create(bufSize, CV_8U);
    if( !mask.empty() )
        maskPyramid.create(bufSize, CV_8U);

    Mat prevImg = image, prevMask = mask;

    // Pre-compute the scale pyramids
    for (level = 0; level < nLevels; ++level)
    {
        Rect linfo = layerInfo[level];
        Size sz(linfo.width, linfo.height);
        Size wholeSize(sz.width + border*2, sz.height + border*2);
        Rect wholeLinfo = Rect(linfo.x - border, linfo.y - border, wholeSize.width, wholeSize.height);
        Mat extImg = imagePyramid(wholeLinfo), extMask;
        Mat currImg = extImg(Rect(border, border, sz.width, sz.height)), currMask;

        if( !mask.empty() )
        {
            extMask = maskPyramid(wholeLinfo);
            currMask = extMask(Rect(border, border, sz.width, sz.height));
        }

        // Compute the resized image
        if( level != firstLevel )
        {
            resize(prevImg, currImg, sz, 0, 0, INTER_LINEAR);
            if( !mask.empty() )
            {
                resize(prevMask, currMask, sz, 0, 0, INTER_LINEAR);
                if( level > firstLevel )
                    threshold(currMask, currMask, 254, 0, THRESH_TOZERO);
            }

            copyMakeBorder(currImg, extImg, border, border, border, border,
                           BORDER_REFLECT_101+BORDER_ISOLATED);
            if (!mask.empty())
                copyMakeBorder(currMask, extMask, border, border, border, border,
                               BORDER_CONSTANT+BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, extImg, border, border, border, border,
                           BORDER_REFLECT_101);
            if( !mask.empty() )
                copyMakeBorder(mask, extMask, border, border, border, border,
                               BORDER_CONSTANT+BORDER_ISOLATED);
        }
        prevImg = currImg;
        prevMask = currMask;
    }

//    if( useOCL ) HOOFR_copyVectorToUMat(layerOfs, ulayerInfo);

    if( do_keypoints )
    {
//        if( useOCL ) imagePyramid.copyTo(uimagePyramid);

        // Get keypoints, those will be far enough from the border that no check will be required for the descriptor
        HOOFR_computeKeyPoints(imagePyramid, uimagePyramid, maskPyramid,
                         layerInfo, ulayerInfo, layerScale, keypoints,
                         nfeatures, scaleFactor, edgeThreshold, patchSize, scoreType, useOCL, fastThreshold);
    }
    else
    {
        KeyPointsFilter::runByImageBorder(keypoints, image.size(), edgeThreshold);

        if( !sortedByLevel )
        {
            std::vector<std::vector<KeyPoint> > allKeypoints(nLevels);
            nkeypoints = (int)keypoints.size();
            for( i = 0; i < nkeypoints; i++ )
            {
                level = keypoints[i].octave;
                CV_Assert(0 <= level);
                allKeypoints[level].push_back(keypoints[i]);
            }
            keypoints.clear();
            for( level = 0; level < nLevels; level++ )
                std::copy(allKeypoints[level].begin(), allKeypoints[level].end(), std::back_inserter(keypoints));
        }
    }

    if( do_descriptors )
    {
        int dsize = descriptorSize();

        nkeypoints = (int)keypoints.size();
        if( nkeypoints == 0 )
        {
            _descriptors.release();
            return;
        }

        _descriptors.create(nkeypoints, dsize, CV_8U);
//        std::vector<Point> pattern;
/*
        const int npoints = 512;
        Point patternbuf[npoints];
        const Point* pattern0 = (const Point*)HOOFR_bit_pattern_31_;

        if( patchSize != 31 )
        {
            pattern0 = patternbuf;
            HOOFR_makeRandomPattern(patchSize, patternbuf, npoints);
        }

        CV_Assert( wta_k == 2 || wta_k == 3 || wta_k == 4 );

        if( wta_k == 2 )
            std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
        else
        {
            int ntuples = descriptorSize()*4;
            initializeHoofrPattern(pattern0, pattern, ntuples, wta_k, npoints);
        }
*/
//        for( level = 0; level < nLevels; level++ )
//        {
            // preprocess the resized image
//            Mat workingMat = imagePyramid(layerInfo[level]);

            //boxFilter(working_mat, working_mat, working_mat.depth(), Size(5,5), Point(-1,-1), true, BORDER_REFLECT_101);
//            GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
//        }

//        if( useOCL )
//        {
//            imagePyramid.copyTo(uimagePyramid);
//            std::vector<Vec4i> kptbuf;
//            UMat ukeypoints, upattern;
//            HOOFR_copyVectorToUMat(pattern, upattern);
//            uploadHOOFRKeypoints(keypoints, layerScale, kptbuf, ukeypoints);

//            UMat udescriptors = _descriptors.getUMat();
//            useOCL = HOOFR_ocl_computeHoofrDescriptors(uimagePyramid, ulayerInfo,
//                                               ukeypoints, udescriptors, upattern,
//                                               nkeypoints, dsize, wta_k);
//            if(useOCL)
//            {
//                CV_IMPL_ADD(CV_IMPL_OCL);
//            }
//        }

//        if( !useOCL )
//        {
            Mat descriptors = _descriptors.getMat();
            //computeHoofrDescriptors(imagePyramid, layerInfo, layerScale, keypoints, descriptors, pattern, dsize, wta_k);
            //////////////
            computeHOOFRdescriptor(image, keypoints, descriptors);
            //////////////
//        }
    }
}

Ptr<HOOFR> HOOFR::create(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold,
           int firstLevel, int wta_k, int scoreType, int patchSize, int fastThreshold)
{
    return makePtr<HOOFR_Impl>(nfeatures, scaleFactor, nlevels, edgeThreshold,
                             firstLevel, wta_k, scoreType, patchSize, fastThreshold);
}

}

