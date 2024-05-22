#include <stdio.h>
void test();
int m = 22, n = 44;
int main()
{
	m = 1;
	n = 2;
 	printf("m=%d, n=%d\n", m, n);
	test();
}
void test()
{
	m = 5, n = 6;
	printf("m=%d, n=%d\n", m, n);
}