import io.shiftleft.semanticcpg.language._
importCpg("/home/kevin/joern-parse/joern-export-demo/cpg.bin")

// 创建导出目录，如果不存在的话
val outputDir = "/home/kevin/joern-parse/joern-export-demo/dot_c/"
new java.io.File(outputDir).mkdirs()

// 用于存储线程函数和其调用者的关系
var threadRelations = Map[String, String]()

// 首先找出所有pthread_create调用及其线程函数
cpg.call("pthread_create").foreach { createCall =>
  // pthread_create的第三个参数是线程函数
  createCall.argument(2) match {
    case arg: Expression =>
      val threadFuncName = arg.code.toString
      val callerMethod = createCall.method.name
      threadRelations += (threadFuncName -> callerMethod)
  }
}

// 遍历每个函数
cpg.method.foreach { method =>
  if (!method.name.startsWith("_") && 
      !method.name.startsWith("pthread") && 
      !method.name.startsWith("perror") && 
      !method.name.startsWith("printf") &&
      !method.name.toLowerCase.contains("global") &&
      !method.name.toLowerCase.contains("operator")) {
    
    val methodName = method.name.replaceAll("[^a-zA-Z0-9]", "_")
    
    // 检查是否是线程函数或其调用者
    val relatedMethod = threadRelations.find(relation => 
      relation._1 == method.name || relation._2 == method.name
    )
    
    relatedMethod match {
      case Some((threadFunc, caller)) =>
        // 这是一个并发相关的函数,需要特殊处理
        val dotFilePath = outputDir + caller + "_with_" + threadFunc + ".dot"
        
        // 获取调用者函数的CPG
        val callerCpg = cpg.method.name(caller).dotCpg14.l
        // 获取线程函数的CPG
        val threadCpg = cpg.method.name(threadFunc).dotCpg14.l
        
        // 合并两个CPG
        (callerCpg, threadCpg) match {
          case (List(callerDot: String), List(threadDot: String)) =>
            // 在这里处理同步块的标记
            val mergedDot = mergeDotGraphs(callerDot, threadDot)
            val writer = new java.io.PrintWriter(dotFilePath)
            writer.write(mergedDot)
            writer.close()
            println(s"Exported merged CPG for concurrent functions to $dotFilePath")
          case _ =>
            println(s"Failed to export merged CPG for concurrent functions")
        }
        
      case None =>
        // 普通函数,按原来的方式处理
        val dotFilePath = outputDir + methodName + ".dot"
        method.dotCpg14.l match {
          case List(dotString: String) =>
            val writer = new java.io.PrintWriter(dotFilePath)
            writer.write(dotString)
            writer.close()
            println(s"Exported CPG for function $methodName to $dotFilePath")
          case _ =>
            println(s"Failed to export CPG for function $methodName")
        }
    }
  }
}

