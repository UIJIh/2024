#include <stdio.h>
int main()
{
    for (int i = 6; i > 0; i--) {
        for (int j = 1; j < i; j++) {
            printf("***");
        }
        printf("\n");
    }
    return 0;
}