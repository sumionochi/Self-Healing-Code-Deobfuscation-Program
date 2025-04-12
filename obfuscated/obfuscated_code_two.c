#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10

// Sums the elements of an array.
int s(int *a, int n) {
    int i, t = 0;
    for (i = 0; i < n; i++) {
        t += a[i];
    }
    return t;
}

// Sorts the array using bubble sort.
void o(int *a, int n) {
    int i, j, t;
    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j+1]) {
                t = a[j];
                a[j] = a[j+1];
                a[j+1] = t;
            }
        }
    }
}

// A simple structure
struct st {
    int a;
    float b;
};

// Modifies a structure and returns a new allocated structure.
struct st *m(struct st *p, int k) {
    struct st *q = (struct st *)malloc(sizeof(struct st));
    if (!q) return NULL;
    q->a = p->a + k;
    q->b = p->b * k;
    return q;
}

int main() {
    int i, x = 0, y = N;
    int *a = (int *)malloc(N * sizeof(int));
    if (!a) {
        printf("Memory allocation error.\n");
        return 1;
    }
    srand(time(NULL));
    
    // Populate the array with random numbers.
    for (i = 0; i < N; i++) {
        a[i] = rand() % 100;
    }
    
    // Calculate the sum of the array.
    x = s(a, N);
    
    // Sort the array.
    o(a, N);
    
    printf("Sum = %d\n", x);
    printf("Sorted array: ");
    for (i = 0; i < N; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
    
    // Demonstrate structure manipulation.
    struct st s1 = { 5, 2.5 };
    struct st *s2 = m(&s1, 3);
    if (s2) {
        printf("Structure: a = %d, b = %f\n", s2->a, s2->b);
        free(s2);
    }
    
    free(a);
    return 0;
}
