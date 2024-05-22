#include <stdio.h>
int main()
{
	printf("char: %d %d\n", sizeof(char), sizeof(unsigned char));
	printf("short: %d %d\n", sizeof(short), sizeof(unsigned short));
	printf("int: %d %d\n", sizeof(int), sizeof(200));
	printf("long: %d %d\n", sizeof(long), sizeof(300L));
	printf("long long: %d %d\n", sizeof(long long), sizeof(900LL));
	printf("float: %d %d\n", sizeof(float), sizeof(3.14F));
	printf("double: %d %d\n", sizeof(double), sizeof(3.14));
	return 0;
}