#include <stdio.h>

int main(void)
{
	int n1 = 7, n2 = 5;
	n1 += n2; //n1 = n1 + n2;
	printf("n1 += n2 : %d\n", n1);
	n2 += 12; //n2 = n2 + 12;
	printf("n2 += 12 : %d\n", n2);
	return 0;
}