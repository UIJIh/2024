#include <stdio.h>
#define HIT_NUM 7
#define _CRT_SECURE_NO_WARNINGS
int main()
{
#if HIT_NUM == 5
	puts("HIT_NUM is 5");
#elif HIT_NUM == 6
	puts("HIT_NUM is 6");
#elif HIT_NUM == 7
	puts("HIT_NUM is 7");
#else
	puts("HIT_NUM is not 5, 6, and 7");
#endif
	return 0;
}