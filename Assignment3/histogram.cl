
__kernel void
set_array_to_constant(
	__global int *array,
	int num_elements,
	int val
)
{
	// There is no need to touch this kernel
	if(get_global_id(0) < num_elements)
		array[get_global_id(0)] = val;
}

__kernel void
compute_histogram(
	__global int *histogram,   // accumulate histogram here
	__global const float *img, // input image
	int width,                 // image width
	int height,                // image height
	int pitch,                 // image pitch
	int num_hist_bins          // number of histogram bins
)
{
	// Insert your kernel code here
	uint X = get_global_id(0);
	uint Y = get_global_id(1);
	int binning = img[Y * pitch + X] * num_hist_bins;

	if (X >= width || Y >= height)
	{
		return;
	}

	//Clamp binning to range 0 to num_hist_bins - 1
	binning = clamp(binning, 0, num_hist_bins - 1);
	atomic_inc(&histogram[binning]);
} 

__kernel void
compute_histogram_local_memory(
	__global int *histogram,   // accumulate histogram here
	__global const float *img, // input image
	int width,                 // image width
	int height,                // image height
	int pitch,                 // image pitch
	int num_hist_bins,         // number of histogram bins
	__local int *local_hist
)
{
	// Insert your kernel code here
	uint X = get_global_id(0);
	uint Y = get_global_id(1);
	uint LID = get_local_id(1) * get_local_size(0) + get_local_id(0);
	int binning = 0;

	if (LID < num_hist_bins)
	{
		local_hist[LID] = 0;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	if (X < width && Y < height) 
	{
		binning = img[Y * pitch + X] * num_hist_bins;
		binning = clamp(binning, 0, num_hist_bins - 1);
		atomic_inc(&local_hist[binning]);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (LID < num_hist_bins)
	{
		atomic_add(&histogram[LID], local_hist[LID]);
	}
		
} 
