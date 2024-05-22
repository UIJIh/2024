#include <stdio.h>
#define SQUARE(x) x*x

int main()
{
	int num = 20;
	printf("Square of num : %d\n", SQUARE(num));
	printf("Square of -5 : %d\n", SQUARE(-5));
	printf("Square of 2.5: %f\n", SQUARE(2.5));
	return 0;
}