#include <stdio.h>
#define DIFF_ABS(x, y) ((x)>(y) ? (x)-(y) : (y)-(x))

int main()
{
	printf("difference: %d\n", DIFF_ABS(5, 7));
	printf("difference: %f\n", DIFF_ABS(1.8, -1.4));
	return 0;
}