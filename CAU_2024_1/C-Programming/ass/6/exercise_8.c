#include <stdio.h>

int main() {

    int x, y;

    printf("Enter two integers (1~100): ");
    scanf_s("%d %d", &x, &y);

    if (1 <= x && x <= 100 && 1 <= y && y <= 100) {
        printf("%d + %d = %d\n", x, y, x + y);
        printf("%d - %d = %d\n", x, y, x - y);
        printf("%d * %d = %d\n", x, y, x * y);
        printf("%d / %d = %d\n", x, y, x / y);
        printf("%d %% %d = %d\n", x, y, x % y);
    }
    else {
        printf("invalid input");
    }

    return 0;
}
