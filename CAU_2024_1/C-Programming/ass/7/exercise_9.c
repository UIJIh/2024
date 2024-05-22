#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

int calculator(int a, char b, int c)
{
    switch (b) {
    case '+':
        return a + c;
    case '-':
        return a - c;
    case '*':
        return a * c;
    case '/':
        if (c != 0) { // 0으로 나누기를 방지
            return a / c;
        }
        else {
            return -1; // 0으로 나누기 시도 시 -1 반환
        }
    case '%':
        if (c != 0) { // 0으로 나누기를 방지
            return a % c;
        }
        else {
            return -1; // 0으로 나누기 시도 시 -1 반환
        }
    default: // 산술연산자만 취급함
        return -1; // 유효하지 않은 연산자의 경우 -1 반환
    }
}

int main(void)
{
    int a, b, result;
    char c;
    scanf("%d %c %d", &a, &c, &b);
    result = calculator(a, c, b);

    printf("%d %c %d = %d\n", a, c, b, result);
    return 0;
}
