# print(result['stdout'])
from cpgqls_client import export_cpg_as_dot, CPGQLSClient

server_endpoint = "localhost:8080"
basic_auth_credentials = ("username", "password")
client = CPGQLSClient(server_endpoint, auth_credentials=basic_auth_credentials)

# execute an `importCode` CPGQuery
posix_code = """
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
"""

openmp_code = """
void main()
{
  int i;
  int myval;
  #pragma omp parallel for ordered
  for (i = 1; i <= 10; i++)
  {
    myval = i + 2;
    #pragma omp ordered
    printf("%d %d\n", i, myval);
  }
}
"""

normal_code = """
void foo () {
  int x = source();
  if(x < MAX) {
	int y = 2*x;
	sink(y);
  }
}
"""

thread_ast = """
void increment(int& shared_var, std::mutex& mtx) {
	std::lock_guard<std::mutex> lock(mtx);
  ++shared_var;
}

int main() {
	int shared_var = 0;
  std::mutex mtx;
  
  std::thread t1 = std::thread(increment, std::ref(shared_var), std::ref(mtx));
  std::thread t2 = std::thread(increment, std::ref(shared_var), std::ref(mtx));
  
  t1.join();
  t2.join();
  
  std::cout << "Final value: " << shared_var << std::endl;
  
  return 0;
}
"""

query = f'importCode.c.fromString("""{thread_ast}""")'
result = client.execute(query)
print(result['stdout'])

# export_cpg_as_json(client, cpg_type="Cpg14")
# export_cpg_as_dot(client, dot_type="Cfg")
export_cpg_as_dot(client, dot_type="Cpg14")