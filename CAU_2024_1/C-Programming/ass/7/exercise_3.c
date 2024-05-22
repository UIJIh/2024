#include <stdio.h>
int Factorial(int n)
{
	int fact = 1;
	for (n; n > 0; n--) {
		fact *= n;
	}
	return fact;
}
int main(void)
{
	printf("1!= %d\n", Factorial(1));
	printf("2!= %d\n", Factorial(2));
	printf("3!= %d\n", Factorial(3));
	printf("4!= %d\n", Factorial(4));
	printf("9!= %d\n", Factorial(9));
	return 0;
}