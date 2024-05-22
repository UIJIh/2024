#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

int main(void)
{
	int a = 1;
	int b = 0;
	if (a || (b = 5)) {
		printf("b is %d\n", b);
	}
	if (a && (b = 5)) {
		printf("b is %d\n", b);
	}
	return 0;
}