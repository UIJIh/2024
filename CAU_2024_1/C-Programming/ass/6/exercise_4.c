#include <stdio.h>
#define ADD 1
#define MIN 0
#define _CRT_SECURE_NO_WARNINGS

int main() {
	int num1, num2;
	printf("enter two integers : ");
	scanf_s("%d %d", &num1, &num2);
#if ADD
	printf("%d + %d = %d\n", num1, num2, num1 + num2);
#endif
#if MIN
	printf("%d - %d = %d\n", num1, num2, num1 - num2);
#endif
	return 0;
}