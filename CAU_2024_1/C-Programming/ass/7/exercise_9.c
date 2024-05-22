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
        if (c != 0) { // 0���� �����⸦ ����
            return a / c;
        }
        else {
            return -1; // 0���� ������ �õ� �� -1 ��ȯ
        }
    case '%':
        if (c != 0) { // 0���� �����⸦ ����
            return a % c;
        }
        else {
            return -1; // 0���� ������ �õ� �� -1 ��ȯ
        }
    default: // ��������ڸ� �����
        return -1; // ��ȿ���� ���� �������� ��� -1 ��ȯ
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
