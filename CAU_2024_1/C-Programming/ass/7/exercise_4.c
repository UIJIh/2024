#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

static int total = 0;

int AddToTotal(int num)
{
    //static int total = 0;
    total += num;
    return total;
}

int main(void)
{
    int num, i;
    for (i = 0; i < 3; i++)
    {
        printf("enter %d:", i + 1);
        scanf("%d", &num);
        printf("total: %d\n", AddToTotal(num));
    }
    //printf("%d", total); // 함수안에서 하면 undeclared identifier
    return 0;
}
