#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

void swap_func(int *x, int *y)
{
	int temp = *x;
	*x = *y;
	*y = temp;
}

int main(void)
{
	int a, b;
	a = 5;
	b = 3;
	printf("before swap : a = %d, b = %d\n", a, b);

	swap_func(&a, &b);

	printf("after swap : a = %d, b = %d\n", a, b);

	return 0;
}
