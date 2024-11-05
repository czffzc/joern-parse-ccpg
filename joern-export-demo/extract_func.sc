importCpg("/home/kevin/joern-parse/joern-export-demo/cpg.bin")

// 创建导出目录，如果不存在的话
val outputDir = "/home/kevin/joern-parse/joern-export-demo/dot/"
new java.io.File(outputDir).mkdirs()

// 遍历每个函数
cpg.method.foreach { method =>
  // 过滤掉不需要的函数
  if (!method.name.startsWith("_") && 
      !method.name.startsWith("pthread") && 
      !method.name.startsWith("perror") && 
      !method.name.startsWith("printf") &&
      !method.name.toLowerCase.contains("global") &&  // 不区分大小写检查global
      !method.name.toLowerCase.contains("operator")) {  // 不区分大小写检查operator
    
    val methodName = method.name.replaceAll("[^a-zA-Z0-9]", "_")  // 清理文件名，避免非法字符
    val dotFilePath = outputDir + methodName + ".dot"

    // 检查是否是线程函数或其调用者
    val relatedMethod = threadRelations.find(relation => 
      relation._1 == method.name || relation._2 == method.name
    )
    
    relatedMethod match {
      case Some((threadFunc, caller)) =>
        // 这是一个并发相关的函数,需要特殊处理
        val dotFilePath = outputDir + caller + "_with_" + threadFunc + ".dot"
        
        // 获取调用者函数的CPG并打印调试信息
        println(s"Looking for caller method: $caller")
      case None =>
        // 普通函数处理...
    }
  }
}