
//Each thread load exactly one halo pixel
//Thus, we assume that the halo size is not larger than the 
//dimension of the work-group in the direction of the kernel

//to efficiently reduce the memory transfer overhead of the global memory
// (each pixel is lodaded multiple times at high overlaps)
// one work-item will compute RESULT_STEPS pixels

//for unrolling loops, these values have to be known at compile time

/* These macros will be defined dynamically during building the program

#define KERNEL_RADIUS 2

//horizontal kernel
#define H_GROUPSIZE_X		32
#define H_GROUPSIZE_Y		4
#define H_RESULT_STEPS		2

//vertical kernel
#define V_GROUPSIZE_X		32
#define V_GROUPSIZE_Y		16
#define V_RESULT_STEPS		3

*/

//#define H_RESULT_STEPS		8

#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

//implemented macros
#define SRC_IMAGE_HORIZONTAL(x,y) (((x) >= 0 && (x) < Width) ? d_Src[(y) * Pitch + (x)] : 0)
#define SRC_IMAGE_VERTICAL(x,y) (((y) >= 0 && (y) < Height) ? d_Src[(y) * Pitch + (x)] : 0)


//////////////////////////////////////////////////////////////////////////////////////////////////////
// Horizontal convolution filter

/*
c_Kernel stores 2 * KERNEL_RADIUS + 1 weights, use these during the convolution
*/

//require matching work-group size
__kernel __attribute__((reqd_work_group_size(H_GROUPSIZE_X, H_GROUPSIZE_Y, 1)))
void ConvHorizontal(
			__global float* d_Dst,
			__global const float* d_Src,
			__constant float* c_Kernel,
			int Width,
			int Pitch
			)
{
	//The size of the local memory: one value for each work-item.
	//We even load unused pixels to the halo area, to keep the code and local memory access simple.
	//Since these loads are coalesced, they introduce no overhead, except for slightly redundant local memory allocation.
	//Each work-item loads H_RESULT_STEPS values + 2 halo values
	__local float tile[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];

	// TODO:
	//const int baseX = ...
	//const int baseY = ...
	//const int offset = ...
	// Load left halo (check for left bound)
	// Load main data + right halo (check for right bound)
	// for (int tileID = 1; tileID < ...)
	// Sync the work-items after loading
	// Convolve and store the result

	uint LIDX = get_local_id(0);
	uint LIDY = get_local_id(1);
	uint GroupIDX = get_group_id(0);
	uint GroupIDY = get_group_id(1);
	int X0 = GroupIDX * H_GROUPSIZE_X * H_RESULT_STEPS - H_GROUPSIZE_X;
	int Y0 = GroupIDY * H_GROUPSIZE_Y;
	int X = 0;
	float temp = 0;
	

	// Load left halo (check for left bound)
	// Load main data + right halo (check for right bound)
	#pragma unroll
	for (int tileID = 0; tileID <= H_RESULT_STEPS + 1; tileID++)
	{
		X = LIDX + H_GROUPSIZE_X * tileID;
		tile[LIDY][X] = SRC_IMAGE_HORIZONTAL(X0 + X, Y0 + LIDY);
	}

	// Sync the work-items after loading
	barrier(CLK_LOCAL_MEM_FENCE);

	// Convolve and store the result
	#pragma unroll
	for (int tileID = 1; tileID <= H_RESULT_STEPS; tileID++)
	{
		X = H_GROUPSIZE_X * tileID + LIDX;

		if (X0 + X >= Width)
		{
			continue;
		}

		temp = 0;

		#pragma unroll
		for (int Xoffset = -KERNEL_RADIUS; Xoffset <= KERNEL_RADIUS; Xoffset++)
		{
			temp += tile[LIDY][X + Xoffset] * c_Kernel[Xoffset + KERNEL_RADIUS];
		}

		d_Dst[(Y0 + LIDY) * Pitch + X0 + X] = temp;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertical convolution filter

//require matching work-group size
__kernel __attribute__((reqd_work_group_size(V_GROUPSIZE_X, V_GROUPSIZE_Y, 1)))
void ConvVertical(
			__global float* d_Dst,
			__global const float* d_Src,
			__constant float* c_Kernel,
			int Height,
			int Pitch
			)
{
	__local float tile[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];

	//TO DO:
	// Conceptually similar to ConvHorizontal
	// Load top halo + main data + bottom halo

	// Compute and store results

	uint LIDX = get_local_id(0);
	uint LIDY = get_local_id(1);
	uint GroupIDX = get_group_id(0);
	uint GroupIDY = get_group_id(1);
	int X0 = GroupIDX * V_GROUPSIZE_X;
	int Y0 = GroupIDY * V_GROUPSIZE_Y * V_RESULT_STEPS - V_GROUPSIZE_Y;
	int Y = 0;
	float temp = 0;

	//Load halo at the top
	if (V_GROUPSIZE_Y - LIDY <= KERNEL_RADIUS)
	{
		tile[LIDY][LIDX] = SRC_IMAGE_VERTICAL(X0 + LIDX, Y0 + LIDY);
	}
	else
	{
		tile[LIDY][LIDX] = 0;
	}

	//Load main data
	#pragma unroll
	for (int tileID = 1; tileID <= V_RESULT_STEPS; tileID++)
	{
		Y = V_GROUPSIZE_Y * tileID + LIDY;

		tile[Y][LIDX] = SRC_IMAGE_VERTICAL(X0 + LIDX, Y0 + Y);
	}

	//Load halo at the bottom
	Y = V_GROUPSIZE_Y * (V_RESULT_STEPS + 1) + LIDY;

	if (LIDY <= KERNEL_RADIUS)
	{
		tile[Y][LIDX] = SRC_IMAGE_VERTICAL(X0 + LIDX, Y0 + Y);
	}
	else
	{
		tile[Y][LIDX] = 0;
	}

	// Sync the work-items after loading
	barrier(CLK_LOCAL_MEM_FENCE);

	// Convolve and store the result
	#pragma unroll
	for (int tileID = 1; tileID <= V_RESULT_STEPS; tileID++)
	{
		Y = LIDY + V_GROUPSIZE_Y * tileID;

		if (Y0 + Y >= Height)
		{
			continue;
		}

		temp = 0;

		#pragma unroll
		for (int Yoffset = -KERNEL_RADIUS; Yoffset <= KERNEL_RADIUS; Yoffset++)
		{
			temp += tile[Y + Yoffset][LIDX] * c_Kernel[Yoffset + KERNEL_RADIUS];
		}

		d_Dst[(Y0 + Y) * Pitch + X0 + LIDX] = temp;
	}
}
