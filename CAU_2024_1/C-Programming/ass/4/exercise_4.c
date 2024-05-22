#include <stdio.h>

int main(void)
{
	int num1, num2;
	num1 = 7;
	num2 = (num1--) + 5;

	printf("num1 : %d\n", num1);
	printf("num2 : %d\n", num2);
	return 0;
}