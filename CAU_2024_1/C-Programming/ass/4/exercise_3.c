#include <stdio.h>

int main(void)
{
	int num1 = 7;
	int num2, num3;

	num2 = num1++;
	num3 = num1--;

	printf("num1 : %d\n", num1);
	printf("num2 : %d\n", num2);
	printf("num3 : %d\n", num3);
	return 0;
}