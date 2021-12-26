/*
We assume a 3x3 (radius: 1) convolution kernel, which is not separable.
Each work-group will process a (TILE_X x TILE_Y) tile of the image.
For coalescing, TILE_X should be multiple of 16.

Instead of examining the image border for each kernel, we recommend to pad the image
to be the multiple of the given tile-size.
*/

//should be multiple of 32 on Fermi and 16 on pre-Fermi...
#define TILE_X 32 

#define TILE_Y 16

//implemented
//returns the float value of the source image at the position of (x,y), checks bounds
#define SRC_IMAGE(x,y) (((x) >= 0 && (x) < Width && (y) >= 0 && (y) < Height) ? d_Src[(y) * Pitch + (x)] : 0)

// d_Dst is the convolution of d_Src with the kernel c_Kernel
// c_Kernel is assumed to be a float[11] array of the 3x3 convolution constants, one multiplier (for normalization) and an offset (in this order!)
// With & Height are the image dimensions (should be multiple of the tile size)
__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1))) //prevent the kernel from running if the size of the work-group is not TILE X * TILE Y.
void Convolution(
				__global float* d_Dst,
				__global const float* d_Src,
				__constant float* c_Kernel,
				uint Width,  // Use width to check for image bounds
				uint Height,
				uint Pitch   // Use pitch for offsetting between lines: preventing bank conflicts by adding a Pad
				)
{
	// OpenCL allows to allocate the local memory from 'inside' the kernel (without using the clSetKernelArg() call)
	// in a similar way to standard C.
	// the size of the local memory necessary for the convolution is the tile size + the halo area
	__local float tile[TILE_X + 2][TILE_Y + 2];

	// TO DO...
	// Fill the halo with zeros
	// Load main filtered area from d_Src
	// Load halo regions from d_Src (edges and corners separately), check for image bounds!
	// Sync threads
	// Perform the convolution and store the convolved signal to d_Dst.

	uint GIDX = get_global_id(0);
	uint GIDY = get_global_id(1);
	uint LIDX = get_local_id(0);
	uint LIDY = get_local_id(1);
	int X = LIDX + 1, Y = LIDY + 1;
	int Xnew = 0, Ynew = 0;
	float temp = 0;

	// Load main filtered area from d_Src
	tile[X][Y] = d_Src[GIDY * Pitch + GIDX];

	// Load halo regions from d_Src (edges and corners separately), check for image bounds!
	if (LIDX == 0)
	{
		tile[0][Y] = SRC_IMAGE(GIDX - 1, GIDY);

		if (LIDY == 0)
		{
			tile[0][0] = SRC_IMAGE(GIDX - 1, GIDY - 1);
		}
		else if (LIDY == TILE_Y - 1)
		{
			tile[0][TILE_Y + 1] = SRC_IMAGE(GIDX - 1, GIDY + 1);
		}
	}
	else if (LIDX == TILE_X - 1)
	{
		tile[TILE_X + 1][Y] = SRC_IMAGE(GIDX + 1, GIDY);

		if (LIDY == 0)
		{
			tile[TILE_X + 1][0] = SRC_IMAGE(GIDX + 1, GIDY - 1);
		}
		else if (LIDY == TILE_Y - 1)
		{
			tile[TILE_X + 1][TILE_Y + 1] = SRC_IMAGE(GIDX + 1, GIDY + 1);
		}
	}

	if (LIDY == 0)
	{
		tile[X][0] = SRC_IMAGE(GIDX, GIDY - 1);
	}
	else if (LIDY == TILE_Y - 1)
	{
		tile[X][TILE_Y + 1] = SRC_IMAGE(GIDX, GIDY + 1);
	}


	// Sync threads : waiting for all local memory operations to complete
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform the convolution and store the convolved signal to d_Dst.

	temp = 0;

	if (GIDY < Height && GIDX < Width) //check images borders
	{
		for (int Yoffset = -1; Yoffset < 2; Yoffset++)
		{
			Ynew = Y + Yoffset;
			for (int Xoffset = -1; Xoffset < 2; Xoffset++)
			{
				Xnew = X + Xoffset;
				temp += tile[Xnew][Ynew] * c_Kernel[(1 + Yoffset) * 3 + (1 + Xoffset)];
			}
		}
		d_Dst[GIDY * Pitch + GIDX] = temp * c_Kernel[9] + c_Kernel[10];
	}
}