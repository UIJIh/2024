#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

int recursive_func(int n)
{
    if (n <= 0) return 0; // F(0) = 0
    if (n == 1) return 1; // F(1) = 1
    return recursive_func(n - 1) + recursive_func(n - 2); // F(n) = F(n-1) + F(n-2)
}

int main(void)
{
    int num, result;
    scanf("%d", &num);
    result = recursive_func(num);

    printf("%d th Fibonacci number = %d\n", num, result);
    return 0;
}
