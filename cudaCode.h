#include <ros/ros.h>
// CUDA-C includes
#include <cufft.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;

void cudaRowReduce();
namespace cr {

class cudaCode {
	private :
	cudaEvent_t start;
	cudaEvent_t stop;

	public:

	cudaCode();
	~cudaCode();
	};

}
