#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

void *threadFunction(void *arg)
{
    int threadNum = *(int *)arg;
    printf("Hello from thread %d\n", threadNum);
    return NULL;
}

int main()
{
    pthread_t threads[2];
    int threadArgs[2];

    for (int i = 0; i < 2; i++)
    {
        threadArgs[i] = i + 1;
        if (pthread_create(&threads[i], NULL, threadFunction, &threadArgs[i]) != 0)
        {
            perror("Failed to create thread");
            return 1;
        }
    }

    for (int i = 0; i < 2; i++)
    {
        if (pthread_join(threads[i], NULL) != 0)
        {
            perror("Failed to join thread");
            return 1;
        }
    }

    printf("All threads are done!\n");
    return 0;
}
