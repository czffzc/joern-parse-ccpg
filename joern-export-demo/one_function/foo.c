// 线程执行的函数
void *threadFunction(void *arg)
{
    int threadNum = *(int *)arg;
    printf("Hello from thread %d\n", threadNum);
    return NULL;
}
