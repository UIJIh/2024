//#include <stdio.h>
//
//int main() {
//
//    int cnt;
//    int x, y;
//
//    printf("Enter number of operation time: ");
//    scanf_s("%d", &cnt);
//    
//    for (int i = 1; i <= cnt; i++) {
//        printf("Enter two integers: ");
//        scanf_s("%d %d", &x, &y);
//           
//        printf("Operation #%d: %d + %d = %d\n", i, x, y, x + y);
//    }
//    return 0;
//}
#include <stdio.h>

#define MAX_OPERATIONS 100 // arbitary decided

int main() {

    int operations[MAX_OPERATIONS][2]; 
    int num_oper;

    printf("Enter the number of operation time: ");
    scanf_s("%d", &num_oper);

    printf("Enter two integers:\n");
    for (int i = 0; i < num_oper; i++) {
        printf("Operation #%d: ", i + 1);
        scanf_s("%d %d", &operations[i][0], &operations[i][1]);
    }

    for (int i = 0; i < num_oper; i++) {
        int x = operations[i][0];
        int y = operations[i][1];
        int result = x + y;
        printf("Operation #%d: %d + %d = %d\n", i + 1, x, y, result);
    }

    return 0;
}
