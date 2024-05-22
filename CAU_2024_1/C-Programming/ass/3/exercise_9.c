#include <stdio.h>
int main()
{
	short a = 4;
	int b = 15;
	printf("a + b = %d, size is %d\n", a+b, sizeof(a+b));
	return 0;
}