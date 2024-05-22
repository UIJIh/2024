#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int main(void)
{
	int a, b, c;
	scanf("%d %d", &a, &b);

	c = (a > b) ? a : b;

	printf("bigger one is %d \n", c);
	return 0;
}