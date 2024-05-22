#include <stdio.h>

int main() {

    int x;
    int sum = 0;
    printf("Enter one integers (1~): ");
    scanf_s("%d", &x);
    for (int i = 1; i <= x; i++) {
        sum += i;
    }
    printf("%d\n", sum);
    return 0;
}
