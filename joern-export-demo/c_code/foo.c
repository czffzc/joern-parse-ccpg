#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

// 线程执行的函数
void* threadFunction(void* arg) {
    int threadNum = *(int*)arg;  // 将参数转换为整数类型
    printf("Hello from thread %d\n", threadNum);
    return NULL;
}

int main() {
    pthread_t threads[2];  // 创建两个线程
    int threadArgs[2];

    // 创建两个线程
    for (int i = 0; i < 2; i++) {
        threadArgs[i] = i + 1;  // 为每个线程传递一个参数
        if (pthread_create(&threads[i], NULL, threadFunction, &threadArgs[i]) != 0) {
            perror("Failed to create thread");
            return 1;
        }
    }

    // 等待所有线程完成
    for (int i = 0; i < 2; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("Failed to join thread");
            return 1;
        }
    }

    printf("All threads are done!\n");
    return 0;
}
