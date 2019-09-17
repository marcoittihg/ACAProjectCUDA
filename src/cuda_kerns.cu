//////////// WRITE YOUR CODE HERE /////////////////
#include <stdio.h>
#include <unistd.h> 
#include <timer.h>
#include <cuda_profiler_api.h>

#define radian_coef 0.024543693
#define grid_factor_d  0.5

#define n_atoms 64

__constant__ int n_rot;
__constant__ int _precision;

texture<float, 3, cudaReadModeElementType> tex3d;

struct Score{
	int index;
	float score;
};

struct RotMatrix{
	float m11;
	float m12;
	float m13;
	float m21;
	float m22;
	float m23;
	float m31;
	float m32;
	float m33;
};

__device__ void warpReduceSum(volatile float* scores, int tid){
	scores[tid] += scores[tid+32];
	scores[tid] += scores[tid+16];
	scores[tid] += scores[tid+8];
	scores[tid] += scores[tid+4];
	scores[tid] += scores[tid+2];
	scores[tid] += scores[tid+1];
}


__global__ void sumScores(Score* scores, float* atoms){
	extern __shared__ float tmpScores[];

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	//Calc rot
	RotMatrix rot;

	int anglex = (bx / n_rot) * _precision;
	int angley = (bx % n_rot) * _precision;

	float alpha = anglex * radian_coef;
	float beta = angley * radian_coef;

	float cosAlpha = cos(alpha);
	float sinAlpha = sin(alpha);
	float cosBeta = cos(beta);
	float sinBeta = sin(beta);

	rot.m11 = cosAlpha*cosBeta;
	rot.m12 = sinAlpha*cosBeta;
	rot.m13 = -sinBeta;

	rot.m21 = -sinAlpha;
	rot.m22 = cosAlpha;

	rot.m31 = cosAlpha * sinBeta;
	rot.m32 = sinAlpha * sinBeta;
	rot.m33 = cosBeta;

	//Calc score
	float oldX = atoms[tx];
	float oldY = atoms[tx+ n_atoms];
	float oldZ = atoms[tx+ 2*n_atoms];

	float newX = rot.m11*oldX + rot.m12*oldY + rot.m13*oldZ;
	float newY = rot.m21*oldX + rot.m22*oldY + rot.m23*oldZ;
	float newZ = rot.m31*oldX + rot.m32*oldY + rot.m33*oldZ;

	newX*= grid_factor_d;
	newY*= grid_factor_d;
	newZ*= grid_factor_d;

	tmpScores[tx] = tex3D(tex3d, newX, newY, newZ);

	__syncthreads();

	if(tx<32){
		warpReduceSum(tmpScores,tx);
	}
	
	__syncthreads();

	if(tx == 0) {
		scores[bx].index = bx;
		scores[bx].score = tmpScores[0];
	}
}


template<unsigned int blockSize>
__device__ void warpReduceMax(volatile Score* scores, int tid){
	if(blockSize>=64){
		if(scores[tid].score < scores[tid+32].score){
			scores[tid].score = scores[tid+32].score;
			scores[tid].index = scores[tid+32].index;
		}
	}
	if(blockSize>=32){
		if(scores[tid].score < scores[tid+16].score){
			scores[tid].score = scores[tid+16].score;
			scores[tid].index = scores[tid+16].index;
		}
	}
	if(blockSize>=16){
		if(scores[tid].score < scores[tid+8].score){
			scores[tid].score = scores[tid+8].score;
			scores[tid].index = scores[tid+8].index;
		}
	}
	if(blockSize>=8){
		if(scores[tid].score < scores[tid+4].score){
			scores[tid].score = scores[tid+4].score;
			scores[tid].index = scores[tid+4].index;
		}
	}
	if(blockSize>=4){
		if(scores[tid].score < scores[tid+2].score){
			scores[tid].score = scores[tid+2].score;
			scores[tid].index = scores[tid+2].index;
		}
	}
	if(blockSize>=2){
		if(scores[tid].score < scores[tid+1].score){
			scores[tid].score = scores[tid+1].score;
			scores[tid].index = scores[tid+1].index;
		}
	}
}

template<unsigned int blockSize>
__global__ void reduceScores(Score* scores, float* inAtoms, float* outAtoms){

	extern __shared__ Score sScores[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (2*blockSize) + tid;
	//unsigned int gridSize = blockSize*2*gridDim.x;
	//printf("gs %d\n", gridSize);

	
	if(scores[i].score > scores[i+blockDim.x].score){
		sScores[tid] = scores[i];
	}else{
		sScores[tid] = scores[i+blockDim.x];
	}

	__syncthreads();


	if(blockSize >= 1024){
		if(tid < 512) {
			if(sScores[tid].score < sScores[tid+512].score){
				sScores[tid] = sScores[tid+512];
			}
		}
		__syncthreads();
	}
	if(blockSize >= 512){
		if(tid < 256) {
			if(sScores[tid].score < sScores[tid+256].score){
				sScores[tid] = sScores[tid+256];
			}
		}
		__syncthreads();
	}
	if(blockSize >= 256){
		if(tid < 128) {
			if(sScores[tid].score < sScores[tid+128].score){
				sScores[tid] = sScores[tid+128];
			}
		}
		__syncthreads();
	}
	if(blockSize >= 128){
		if(tid < 64) {
			if(sScores[tid].score < sScores[tid+64].score){
				sScores[tid] = sScores[tid+64];
			}
		}
		__syncthreads();
	}

	if(tid < 32) warpReduceMax<blockSize>(sScores, tid);

	__syncthreads();
	
	if(tid < n_atoms){
		RotMatrix rot;
		int index = sScores[0].index;
		int anglex = (index / n_rot) * _precision;
		int angley = (index % n_rot) * _precision;
		//printf("angles %d %d\n", anglex, angley);

		float alpha = anglex * radian_coef;
		float beta = angley * radian_coef;

		float cosAlpha = cos(alpha);
		float sinAlpha = sin(alpha);
		float cosBeta = cos(beta);
		float sinBeta = sin(beta);

		rot.m11 = cosAlpha*cosBeta;
		rot.m12 = sinAlpha*cosBeta;
		rot.m13 = -sinBeta;

		rot.m21 = -sinAlpha;
		rot.m22 = cosAlpha;
		rot.m23 = 0;

		rot.m31 = cosAlpha * sinBeta;
		rot.m32 = sinAlpha * sinBeta;
		rot.m33 = cosBeta;

		float oldX = inAtoms[tid];
		float oldY = inAtoms[tid+ n_atoms];
		float oldZ = inAtoms[tid+ 2*n_atoms];

		outAtoms[tid] = rot.m11*oldX + rot.m12*oldY + rot.m13*oldZ;
		outAtoms[tid+n_atoms] = rot.m21*oldX + rot.m22*oldY + rot.m23*oldZ;
		outAtoms[tid+2*n_atoms] = rot.m31*oldX + rot.m32*oldY + rot.m33*oldZ;
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern "C" void align_kern(float* atoms_in, float* atoms_out, int precision, float* score_pos){
	//CONSTANTS
	float ma = static_cast<float>(256);
	int nRot = ma / precision;
	int nAngles = nRot * nRot;
	//printf("\n\n %d \n\n", nAngles);
	

	//Memory
	gpuErrchk(cudaMemcpyToSymbol(n_rot, &nRot, sizeof(n_rot)));
	gpuErrchk(cudaMemcpyToSymbol(_precision, &precision, sizeof(precision)));

	Score* scores;
	gpuErrchk(cudaMalloc((void**)&scores, nAngles * sizeof(Score)));

	float* atoms;
	gpuErrchk(cudaMalloc((void**)&atoms, 3* n_atoms * sizeof(float)));
	gpuErrchk(cudaMemcpy(atoms, atoms_in, 3* n_atoms * sizeof(float), cudaMemcpyHostToDevice));

	float* outAtoms;
	gpuErrchk(cudaMalloc((void**)&outAtoms, 3* n_atoms * sizeof(float)));
	gpuErrchk(cudaMemcpy(atoms, atoms_in, 3* n_atoms * sizeof(float), cudaMemcpyHostToDevice));


	//3D texture
    const int dim = 100;
    float *data = (float *)malloc(dim*dim*dim*sizeof(float));
    for (int z = 0; z < dim; z++)
    	for (int y = 0; y < dim; y++)
        	for (int x = 0; x < dim; x++)
          		data[z*dim*dim+y*dim+x] = score_pos[x+100*y+10000*z];
        	

   	cudaExtent vol = {dim,dim,dim};
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *d_volumeArray = 0;
    gpuErrchk(cudaMalloc3DArray(&d_volumeArray, &channelDesc, vol));

    cudaMemcpy3DParms p = {0};
    p.srcPtr = make_cudaPitchedPtr((void *)data, vol.width*sizeof(float), vol.width, vol.height);
    p.dstArray = d_volumeArray;
    p.extent   = vol;
   	p.kind = cudaMemcpyHostToDevice;
    gpuErrchk(cudaMemcpy3D(&p));

	tex3d.normalized = false; 
    tex3d.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates
    tex3d.addressMode[1] = cudaAddressModeClamp;
    tex3d.addressMode[2] = cudaAddressModeClamp;

    gpuErrchk(cudaBindTextureToArray(tex3d, d_volumeArray, channelDesc));


    //Kernel execution
 	cudaProfilerStart();

	sumScores<<<nAngles,n_atoms, n_atoms * sizeof(float)>>>(scores, atoms);
	reduceScores<1024><<<1,nAngles, nAngles * sizeof(Score)>>> (scores, atoms, outAtoms);

	cudaProfilerStop();

	//D to H memory
	Score* scoresh = (Score*)malloc(nAngles * sizeof(Score));
	gpuErrchk(cudaMemcpy(atoms_out, outAtoms, 3 * n_atoms * sizeof(float), cudaMemcpyDeviceToHost));


	//Free memory
	cudaFree(scores);
	cudaFree(atoms);
	cudaFree(outAtoms);
	cudaFree(d_volumeArray);

	cudaUnbindTexture(tex3d);
}
