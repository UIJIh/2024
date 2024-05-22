#include <stdio.h>
int main(void)
{
	int n1 = +2147483647; //Max value
	int n2 = -2147483647; //Min value
	printf("before overflow : %d\n", n1);
	n1 = n1 + 100; //overflow occurs
	printf("after overflow : %d\n", n1);
	printf("before underflow : %d\n", n2);
	n2 = n2 - 100; //underflow occurs
	printf("after underflow : %d\n", n2);
	return 0;
}