#include <stdio.h>

int main(void)
{
	int x = 3;
	int y = 2;
	printf("% d\n", (x > y) ? x : y);
	printf("% d\n", (x < y) ? x : y);
	return 0;
}