#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

void quiz_game(int a)
{
    int quess;
    while (1) {
        scanf("%d", &quess);
        if (a == quess) {
            printf("correct!!\n");
            return;
        }
        else if (a > quess) {
            printf("answer is bigger than %d\n", quess);
        }
        else {
            printf("answer is smaller than %d\n", quess);
        }
    }
}

int main(void)
{
    int a;
    printf("answer : ");
    scanf("%d", &a);

    printf("quiz start!\n");
    quiz_game(a);
    printf("quiz end!\n");
    return 0;
}
