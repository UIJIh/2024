#include <stdio.h>
int global_variable = 55;
int main()
{
	int local_variable = 44;
	printf("global_variable is %d\n", global_variable);
	printf("local_variable is %d\n", local_variable);
	return 0;
}
