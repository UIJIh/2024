#include <stdio.h>
#include <stdlib.h>

int main() {
    //int* p;
    //p = (int*)malloc(sizeof(int));

    //if (p == NULL) {
    //    printf("Memory allocation failed\n");
    //    return 1;
    //}

    //// 항상 if (p != NULL) 확인해야함!!
    //printf("%p\n", p); // p가 자체 가리키는 메모리 00000295FE6F6A40
    //printf("%d\n", *p); // -842150451(쓰레기값), p가 가리키는 메모리 위치에 저장된 값
    //printf("%d\n", &p); // p 변수 자체의 주소 >> 동적할당 계속 바뀜
    //printf("%d\n", sizeof(*p)); // 4
    //printf("%d\n\n", sizeof(p)); // 8
    //free(p);

    //int* a;
    //a = malloc(sizeof(int));
    //printf("%d\n", *a); // -842150451
    //printf("%d\n", &a); // 계속 바뀐다.
    //printf("%d\n", sizeof(*a)); // 4
    //printf("%d\n\n", sizeof(a)); // 8
    //free(a);

    //int* b;   
    //b = malloc(sizeof(char));
    //*b = 10;
    //printf("%d\n", *b); // 10 값이 할당 됐으니까 쓰레기 값이 아니다
    //printf("%d\n", &b); // 동적 할당, 계속 바뀐다.
    //printf("%d\n", sizeof(*b)); // 4
    //printf("%d\n\n", sizeof(b)); // 8 (포인터 변수의 사이즈, 64bit에서는 포인터 크기가 8byte)
    //free(b);

    int* ptr;
    ptr = calloc(10, sizeof(int)); // malloc(40) 정수 10개 저장가능하고 0으로 초기화
    if (ptr == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    printf("%d\n", *ptr); // 0
    printf("%d\n", *(ptr + 1)); // 0
    printf("%d\n", &ptr); // the address of "the pointer variable ptr itself". 계속 바뀜
    //printf("%d\n", &(ptr + 1)); 이건 안돼! &ptr은 변수 자체의 주소
    printf("%d\n", ptr); // 35823616 가리키는 주소 (allocated, 얘도 계속 바뀜)
    printf("%d\n", ptr+1); // 35823620 가리키는 주소
    printf("%d\n", sizeof(*ptr)); // 4
    printf("%d\n", sizeof(ptr)); // 8
    free(ptr);

    // ptr => |0x1|0x5|..|..|
    // (0x1)|0| (0x5)|0|

    return 0;
}


