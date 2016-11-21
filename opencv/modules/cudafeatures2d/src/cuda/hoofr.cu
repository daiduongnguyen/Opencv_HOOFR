/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if !defined CUDA_DISABLER

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/functional.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace hoofr
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        //buid Pattern            
        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // cull

        int cull_gpu(int* loc, float* response, int size, int n_points)
        {
            thrust::device_ptr<int> loc_ptr(loc);
            thrust::device_ptr<float> response_ptr(response);

            thrust::sort_by_key(response_ptr, response_ptr + size, loc_ptr, thrust::greater<float>());

            return n_points;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // HessianResponses
			__device__ float hessian_dxx[49] =
            {
				0.0008,    0.0015,   -0.0007,   -0.0034,   -0.0007,    0.0015,    0.0008,
				0.0044,    0.0085,   -0.0041,   -0.0191,   -0.0041,    0.0085,    0.0044,
				0.0125,    0.0240,   -0.0117,   -0.0542,   -0.0117,    0.0240,    0.0125,
				0.0177,    0.0340,   -0.0166,   -0.0768,   -0.0166,    0.0340,    0.0177,
				0.0125,    0.0240,   -0.0117,   -0.0542,   -0.0117,    0.0240,    0.0125,
				0.0044,    0.0085,   -0.0041,   -0.0191,   -0.0041,    0.0085,    0.0044,
				0.0008,    0.0015,   -0.0007,   -0.0034,   -0.0007,    0.0015,    0.0008
            };
            
            __device__  float hessian_dyy[49] = 
			{
				0.0008,    0.0044,    0.0125,    0.0177,    0.0125,    0.0044,    0.0008,
				0.0015,    0.0085,    0.0240,    0.0340,    0.0240,    0.0085,    0.0015,
			   -0.0007,   -0.0041,   -0.0117,   -0.0166,   -0.0117,   -0.0041,   -0.0007,
			   -0.0034,   -0.0191,   -0.0542,   -0.0768,   -0.0542,   -0.0191,   -0.0034,
			   -0.0007,   -0.0041,   -0.0117,   -0.0166,   -0.0117,   -0.0041,   -0.0007,
				0.0015,    0.0085,    0.0240,    0.0340,    0.0240,    0.0085,    0.0015,
				0.0008,    0.0044,    0.0125,    0.0177,    0.0125,    0.0044,    0.0008

			}; 
			__device__  float hessian_dxy[49] = 
			{
				0.0009,    0.0035,    0.0050,         0,   -0.0050,   -0.0035,   -0.0009,
				0.0035,    0.0133,    0.0188,         0,   -0.0188,   -0.0133,   -0.0035,
				0.0050,    0.0188,    0.0266,         0,   -0.0266,   -0.0188,   -0.0050,
					 0,         0,         0,         0,         0,         0,         0,
			   -0.0050,   -0.0188,   -0.0266,         0,    0.0266,    0.0188,    0.0050,
			   -0.0035,   -0.0133,   -0.0188,         0,    0.0188,    0.0133,    0.0035,
			   -0.0009,   -0.0035,   -0.0050,         0,    0.0050,    0.0035,    0.0009
			}; 


        __global__ void HessianResponses(const PtrStepb img, const short2* loc_, float* response, const int npoints, const int blockSize, const float hessian_k)
        {
                   
            __shared__ float smem0[8 * 32];
            __shared__ float smem1[8 * 32];
            __shared__ float smem2[8 * 32];

            const int ptidx = blockIdx.x * blockDim.y + threadIdx.y;

            if (ptidx < npoints)
            {
                const short2 loc = loc_[ptidx];

                const int r = blockSize / 2;
                const int x0 = loc.x - r;
                const int y0 = loc.y - r;

                float Dxx = 0, Dyy = 0, Dxy = 0;

                for (int ind = threadIdx.x; ind < blockSize * blockSize; ind += blockDim.x)
                {
                    const int i = ind / blockSize;
                    const int j = ind % blockSize;         

                    Dxx += ((float)img(y0 + i, x0 + j)) * hessian_dxx[ind]; 
                    Dyy += ((float)img(y0 + i, x0 + j)) * hessian_dyy[ind]; 
                    Dxy += ((float)img(y0 + i, x0 + j)) * hessian_dxy[ind]; 
                }

                float* srow0 = smem0 + threadIdx.y * blockDim.x;
                float* srow1 = smem1 + threadIdx.y * blockDim.x;
                float* srow2 = smem2 + threadIdx.y * blockDim.x;

                plus<float> op;
                reduce<32>(smem_tuple(srow0, srow1, srow2), thrust::tie(Dxx, Dyy, Dxy), threadIdx.x, thrust::make_tuple(op, op, op));

                if (threadIdx.x == 0)
                {                
                    response[ptidx] = (Dxx * Dyy) - hessian_k*(Dxy * Dxy);                   
                }
            }
        }

        void HessianResponses_gpu(PtrStepSzb img, const short2* loc, float* response, const int npoints, int blockSize, float hessian_k, cudaStream_t stream)
        {
            dim3 block(32, 8);

            dim3 grid;
            grid.x = divUp(npoints, block.y);

            HessianResponses<<<grid, block, 0, stream>>>(img, loc, response, npoints, blockSize, hessian_k);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // IC_Angle

        __constant__ int c_u_max[32];

        void loadUMax(const int* u_max, int count)
        {
            cudaSafeCall( cudaMemcpyToSymbol(c_u_max, u_max, count * sizeof(int)) );
        }

        __global__ void IC_Angle(const PtrStepb image, const short2* loc_, float* angle, const int npoints, const int half_k)
        {
            __shared__ int smem0[8 * 32];
            __shared__ int smem1[8 * 32];

            int* srow0 = smem0 + threadIdx.y * blockDim.x;
            int* srow1 = smem1 + threadIdx.y * blockDim.x;

            plus<int> op;

            const int ptidx = blockIdx.x * blockDim.y + threadIdx.y;

            if (ptidx < npoints)
            {
                int m_01 = 0, m_10 = 0;

                const short2 loc = loc_[ptidx];

                // Treat the center line differently, v=0
                for (int u = threadIdx.x - half_k; u <= half_k; u += blockDim.x)
                    m_10 += u * image(loc.y, loc.x + u);

                reduce<32>(srow0, m_10, threadIdx.x, op);

                for (int v = 1; v <= half_k; ++v)
                {
                    // Proceed over the two lines
                    int v_sum = 0;
                    int m_sum = 0;
                    const int d = c_u_max[v];

                    for (int u = threadIdx.x - d; u <= d; u += blockDim.x)
                    {
                        int val_plus = image(loc.y + v, loc.x + u);
                        int val_minus = image(loc.y - v, loc.x + u);

                        v_sum += (val_plus - val_minus);
                        m_sum += u * (val_plus + val_minus);
                    }

                    reduce<32>(smem_tuple(srow0, srow1), thrust::tie(v_sum, m_sum), threadIdx.x, thrust::make_tuple(op, op));

                    m_10 += m_sum;
                    m_01 += v * v_sum;
                }

                if (threadIdx.x == 0)
                {
                    float kp_dir = ::atan2f((float)m_01, (float)m_10);
                    kp_dir += (kp_dir < 0) * (2.0f * CV_PI_F);
                    kp_dir *= 180.0f / CV_PI_F;

                    angle[ptidx] = kp_dir;
                }
            }
        }

        void IC_Angle_gpu(PtrStepSzb image, const short2* loc, float* angle, int npoints, int half_k, cudaStream_t stream)
        {
            dim3 block(32, 8);

            dim3 grid;
            grid.x = divUp(npoints, block.y);

            IC_Angle<<<grid, block, 0, stream>>>(image, loc, angle, npoints, half_k);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        __device__ uchar meanIntensity(const PtrStepi integral, const float kp_x, const float kp_y, const int scale, const int rot, const int point, 
                                       const float* patternLookup_x, const float* patternLookup_y, const float* patternLookup_sigma, const int NB_ORIENTATION, const int NB_POINTS)
        {
			int id = scale * NB_ORIENTATION * NB_POINTS + rot * NB_POINTS + point;
			float xf = patternLookup_x[id] + kp_x;
			float yf = patternLookup_y[id] + kp_y;
			
			float radius = patternLookup_sigma[id];
			
			if(radius < 0.5) {radius = 0.5;}
			
			int x_left = int(xf-radius+0.5);
			int y_top = int(yf-radius+0.5);
			int x_right = int(xf+radius+1.5);
			int y_bottom = int(yf+radius+1.5);
			
			int ret_val;
			
			ret_val = integral(y_bottom,x_right) - integral(y_bottom,x_left);//bottom right corner
			ret_val += integral(y_top,x_left);
			ret_val -= integral(y_top,x_right);
			ret_val = ret_val/( (x_right-x_left) * (y_bottom-y_top) );
			
			return ((uchar)ret_val);
        }
        
        __global__ void computeHOOFRDescriptor(PtrStepSzi imgintegral, float* patternLookup_x, float* patternLookup_y, float* patternLookup_sigma, 
                                        uchar* descriptionPairs_i, uchar* descriptionPairs_j, int* orientationPairs_i, int* orientationPairs_j,
                                        int* orientationPairs_wx, int* orientationPairs_wy, int npoints,const int NB_POINTS, int NB_ORIENTATION, int NB_PAIRS,
                                        float* keypoint_x, float* keypoint_y, float* keypoint_angle, float* keypoint_octave, PtrStepb desc)
        {
			const int ptidx = blockIdx.x * blockDim.y + threadIdx.y;
			if (ptidx < npoints)
			{
				uchar pointsValue[49];			
				int direction0 = 0;
				int direction1 = 0;
			
				for( int i = 0; i < NB_POINTS-9; i++)
				{
					pointsValue[i] = meanIntensity(imgintegral, keypoint_x[ptidx], keypoint_y[ptidx], keypoint_octave[ptidx], 0, i, patternLookup_x, patternLookup_y, patternLookup_sigma, NB_ORIENTATION, NB_POINTS);
				}
			
				for( int m = 0; m < 40; m++)
				{
					//iterate through the orientation pairs
					const int delta = (pointsValue[ orientationPairs_i[m] ] - pointsValue[ orientationPairs_j[m] ]);
					direction0 += delta*(orientationPairs_wx[m])/2048;
					direction1 += delta*(orientationPairs_wy[m])/2048;
				}
            
				keypoint_angle[ptidx] = static_cast<float>(atan2((float)direction1,(float)direction0)*(180.0/CV_PI));	
				
				int thetaIdx = int(NB_ORIENTATION*keypoint_angle[ptidx]*(1/360.0)+0.5);
						
				if( thetaIdx < 0 )
					thetaIdx += NB_ORIENTATION;

				if( thetaIdx >= NB_ORIENTATION )
					thetaIdx -= NB_ORIENTATION;
            
				for( int i = 0; i < NB_POINTS; i++)
				{
					pointsValue[i] = meanIntensity(imgintegral, keypoint_x[ptidx], keypoint_y[ptidx], keypoint_octave[ptidx], thetaIdx, i, patternLookup_x, patternLookup_y, patternLookup_sigma, NB_ORIENTATION, NB_POINTS);
				}
            
				////////////////////
				const int n_word = NB_PAIRS/32; 
				int cnt;
				unsigned int reg;
				for (int i = 0; i < n_word; i++)
				{
					reg = 0;
					for (int j = 0; j < 32; j++)
					{
						cnt = j + i * 32;
						if(pointsValue[descriptionPairs_i[cnt]] >= pointsValue[descriptionPairs_j[cnt]]) { reg |= (1<<j); }						
					}
					unsigned int* r_t = (unsigned int*) (&(desc.ptr(ptidx)[4*i])); 
					*r_t = reg;
				}									
				
			}               
        } 
                
        
        void computeHOOFRDescriptor_gpu(PtrStepSzi imgintegral, float* patternLookup_x, float* patternLookup_y, float* patternLookup_sigma, 
                                        uchar* descriptionPairs_i, uchar* descriptionPairs_j, int* orientationPairs_i, int* orientationPairs_j,
                                        int* orientationPairs_wx, int* orientationPairs_wy, int npoints, int NB_POINTS, int NB_ORIENTATION, int NB_PAIRS,
                                        float* keypoint_x, float* keypoint_y, float* keypoint_angle, float* keypoint_octave, PtrStepb desc, cudaStream_t stream) 
        {
			dim3 block(1, 8);

            dim3 grid;
            grid.x = divUp(npoints, block.y);

            computeHOOFRDescriptor<<<grid, block, 0, stream>>>(imgintegral, patternLookup_x, patternLookup_y, patternLookup_sigma,
                                                               descriptionPairs_i, descriptionPairs_j, orientationPairs_i, orientationPairs_j,
                                                               orientationPairs_wx, orientationPairs_wy, npoints, NB_POINTS, NB_ORIENTATION, NB_PAIRS,
                                                               keypoint_x, keypoint_y, keypoint_angle, keypoint_octave, desc);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // computeHoofrDescriptor

        template <int WTA_K> struct HoofrDescriptor;

        #define GET_VALUE(idx) \
            img(loc.y + __float2int_rn(pattern_x[idx] * sina + pattern_y[idx] * cosa), \
                loc.x + __float2int_rn(pattern_x[idx] * cosa - pattern_y[idx] * sina))

        template <> struct HoofrDescriptor<2>
        {
            __device__ static int calc(const PtrStepb& img, short2 loc, const int* pattern_x, const int* pattern_y, float sina, float cosa, int i)
            {
                pattern_x += 16 * i;
                pattern_y += 16 * i;

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

                return val;
            }
        };

        template <> struct HoofrDescriptor<3>
        {
            __device__ static int calc(const PtrStepb& img, short2 loc, const int* pattern_x, const int* pattern_y, float sina, float cosa, int i)
            {
                pattern_x += 12 * i;
                pattern_y += 12 * i;

                int t0, t1, t2, val;

                t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
                val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

                t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

                t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

                t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

                return val;
            }
        };

        template <> struct HoofrDescriptor<4>
        {
            __device__ static int calc(const PtrStepb& img, short2 loc, const int* pattern_x, const int* pattern_y, float sina, float cosa, int i)
            {
                pattern_x += 16 * i;
                pattern_y += 16 * i;

                int t0, t1, t2, t3, k, val;
                int a, b;

                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                t2 = GET_VALUE(2); t3 = GET_VALUE(3);
                a = 0, b = 2;
                if( t1 > t0 ) t0 = t1, a = 1;
                if( t3 > t2 ) t2 = t3, b = 3;
                k = t0 > t2 ? a : b;
                val = k;

                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                t2 = GET_VALUE(6); t3 = GET_VALUE(7);
                a = 0, b = 2;
                if( t1 > t0 ) t0 = t1, a = 1;
                if( t3 > t2 ) t2 = t3, b = 3;
                k = t0 > t2 ? a : b;
                val |= k << 2;

                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                t2 = GET_VALUE(10); t3 = GET_VALUE(11);
                a = 0, b = 2;
                if( t1 > t0 ) t0 = t1, a = 1;
                if( t3 > t2 ) t2 = t3, b = 3;
                k = t0 > t2 ? a : b;
                val |= k << 4;

                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                t2 = GET_VALUE(14); t3 = GET_VALUE(15);
                a = 0, b = 2;
                if( t1 > t0 ) t0 = t1, a = 1;
                if( t3 > t2 ) t2 = t3, b = 3;
                k = t0 > t2 ? a : b;
                val |= k << 6;

                return val;
            }
        };

        #undef GET_VALUE

        template <int WTA_K>
        __global__ void computeHoofrDescriptor(const PtrStepb img, const short2* loc, const float* angle_, const int npoints,
            const int* pattern_x, const int* pattern_y, PtrStepb desc, int dsize)
        {
            const int descidx = blockIdx.x * blockDim.x + threadIdx.x;
            const int ptidx = blockIdx.y * blockDim.y + threadIdx.y;

            if (ptidx < npoints && descidx < dsize)
            {
                float angle = angle_[ptidx];
                angle *= (float)(CV_PI_F / 180.f);

                float sina, cosa;
                ::sincosf(angle, &sina, &cosa);

                desc.ptr(ptidx)[descidx] = HoofrDescriptor<WTA_K>::calc(img, loc[ptidx], pattern_x, pattern_y, sina, cosa, descidx);
            }
        }

        void computeHoofrDescriptor_gpu(PtrStepb img, const short2* loc, const float* angle, const int npoints,
            const int* pattern_x, const int* pattern_y, PtrStepb desc, int dsize, int WTA_K, cudaStream_t stream)
        {
            dim3 block(32, 8);

            dim3 grid;
            grid.x = divUp(dsize, block.x);
            grid.y = divUp(npoints, block.y);

            switch (WTA_K)
            {
            case 2:
                computeHoofrDescriptor<2><<<grid, block, 0, stream>>>(img, loc, angle, npoints, pattern_x, pattern_y, desc, dsize);
                break;

            case 3:
                computeHoofrDescriptor<3><<<grid, block, 0, stream>>>(img, loc, angle, npoints, pattern_x, pattern_y, desc, dsize);
                break;

            case 4:
                computeHoofrDescriptor<4><<<grid, block, 0, stream>>>(img, loc, angle, npoints, pattern_x, pattern_y, desc, dsize);
                break;
            }

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // mergeLocation

        __global__ void mergeLocation(const short2* loc_, float* x, float* y, const int npoints, float scale)
        {
            const int ptidx = blockIdx.x * blockDim.x + threadIdx.x;

            if (ptidx < npoints)
            {
                short2 loc = loc_[ptidx];

                x[ptidx] = loc.x * scale;
                y[ptidx] = loc.y * scale;
            }
        }

        void mergeLocation_gpu(const short2* loc, float* x, float* y, int npoints, float scale, cudaStream_t stream)
        {
            dim3 block(256);

            dim3 grid;
            grid.x = divUp(npoints, block.x);

            mergeLocation<<<grid, block, 0, stream>>>(loc, x, y, npoints, scale);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    }
}}}

#endif /* CUDA_DISABLER */
