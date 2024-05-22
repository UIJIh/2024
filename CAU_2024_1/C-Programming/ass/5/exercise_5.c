#include <stdio.h>
int main()
{
	int answer = 41;
	int guess;
	int tries = 0;
	do {
		printf("guess the answer : ");
		scanf_s(" %d", &guess);
		tries++;
		if (guess > answer)
			printf("the number is higher than answer\n");
		if (guess < answer)
			printf("the number is lower than answer\n");
	} while (guess != answer);
	printf("tries = %d\n", tries);
	return 0;
}