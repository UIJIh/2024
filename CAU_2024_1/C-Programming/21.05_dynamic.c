#include <stdio.h>
#include <stdlib.h>

int main() {
    //int* p;
    //p = (int*)malloc(sizeof(int));

    //if (p == NULL) {
    //    printf("Memory allocation failed\n");
    //    return 1;
    //}

    //// �׻� if (p != NULL) Ȯ���ؾ���!!
    //printf("%p\n", p); // p�� ��ü ����Ű�� �޸� 00000295FE6F6A40
    //printf("%d\n", *p); // -842150451(�����Ⱚ), p�� ����Ű�� �޸� ��ġ�� ����� ��
    //printf("%d\n", &p); // p ���� ��ü�� �ּ� >> �����Ҵ� ��� �ٲ�
    //printf("%d\n", sizeof(*p)); // 4
    //printf("%d\n\n", sizeof(p)); // 8
    //free(p);

    //int* a;
    //a = malloc(sizeof(int));
    //printf("%d\n", *a); // -842150451
    //printf("%d\n", &a); // ��� �ٲ��.
    //printf("%d\n", sizeof(*a)); // 4
    //printf("%d\n\n", sizeof(a)); // 8
    //free(a);

    //int* b;   
    //b = malloc(sizeof(char));
    //*b = 10;
    //printf("%d\n", *b); // 10 ���� �Ҵ� �����ϱ� ������ ���� �ƴϴ�
    //printf("%d\n", &b); // ���� �Ҵ�, ��� �ٲ��.
    //printf("%d\n", sizeof(*b)); // 4
    //printf("%d\n\n", sizeof(b)); // 8 (������ ������ ������, 64bit������ ������ ũ�Ⱑ 8byte)
    //free(b);

    int* ptr;
    ptr = calloc(10, sizeof(int)); // malloc(40) ���� 10�� ���尡���ϰ� 0���� �ʱ�ȭ
    if (ptr == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    printf("%d\n", *ptr); // 0
    printf("%d\n", *(ptr + 1)); // 0
    printf("%d\n", &ptr); // the address of "the pointer variable ptr itself". ��� �ٲ�
    //printf("%d\n", &(ptr + 1)); �̰� �ȵ�! &ptr�� ���� ��ü�� �ּ�
    printf("%d\n", ptr); // 35823616 ����Ű�� �ּ� (allocated, �굵 ��� �ٲ�)
    printf("%d\n", ptr+1); // 35823620 ����Ű�� �ּ�
    printf("%d\n", sizeof(*ptr)); // 4
    printf("%d\n", sizeof(ptr)); // 8
    free(ptr);

    // ptr => |0x1|0x5|..|..|
    // (0x1)|0| (0x5)|0|

    return 0;
}


