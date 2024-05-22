#include <stdio.h>
#define NAME "Son"
#define AGE 20
#define PRINT_ADDR puts("address: ¡¦\n");
int main()
{
	printf("Name: %s\n", NAME);
	printf("Age: %d\n", AGE);
	PRINT_ADDR;
	return 0;
}