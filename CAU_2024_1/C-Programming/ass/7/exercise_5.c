#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

int var = 0;
int func()
{	
	int var = 3;
	return printf("var : %d\n", var);
}

int main(void)
{
	func();
	printf("var : %d\n", var);
	return 0;
}
