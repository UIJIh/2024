#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
int main(void)
{
	int a, b;
	scanf("%d %d", &a, &b);

	if (a > b) {
		printf("a is bigger than b\n");
	}
	else if (a < b) {
		printf("b is bigger than a\n");
	}

	return 0;
}