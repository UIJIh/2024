#include <stdio.h>

int main(void)
{
	int num = 1;
	if (num == 1) {
		int num = 7;
		num += 10;
		printf("if local variable num : %d\n", num);
	}
	printf("main local variable num : %d\n", num);
	return 0;
}