#include <stdio.h>
int main()
{
	int i;
	int j;
	for (i = 0; i < 2; i++)
	{
		for (j = 0; j < 4; j++)
		{
			printf("%d, %d\n", i, j);
		}
	}
	return 0;
}