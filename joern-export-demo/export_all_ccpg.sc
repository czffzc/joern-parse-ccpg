@main def main(codeDir: String, outputDir: String) = {
  importCode(codeDir)
  exportAllFunctionsToDot(outputDir)
}

def exportAllFunctionsToDot(outputDir: String) = {
  println("\n导出所有函数的合并CPG:")
  println("====================")

  // 初始化合并后的dot字符串
  var combinedDotString = "digraph combined_cpg {\n"

  // 存储所有已处理的节点和边,避免重复
  var processedNodes = Set[String]()
  var processedEdges = Set[String]()

  // 存储需要标记为阻塞的节点ID
  var blockingNodes = Set[Long]()

  // 首先标记所有lock-unlock之间的节点
  cpg.method.foreach { method =>
    val lockNodes = method.call.name(".*lock").l
    val unlockNodes = method.call.name(".*unlock").l

    lockNodes.foreach { lockNode =>
      unlockNodes.foreach { unlockNode =>
        // 获取lock到unlock之间的所有节点
        val nodesInBetween = lockNode.ast
          .takeWhile(node => node.id != unlockNode.id)
          .filter(node => node != lockNode && node != unlockNode)
          .l

        // 添加到阻塞节点集合
        blockingNodes ++= nodesInBetween.map(_.id)
      }
    }
  }

  // 标记pthread_join相关的节点
  cpg.call.name("pthread_join").foreach { joinCall =>
    blockingNodes += joinCall.id
  }

  // 处理每个函数
  cpg.method.foreach { method =>
    // 过滤掉不需要的函数
    if (
      !method.name.startsWith("pthread") &&
      !method.name.startsWith("perror") &&
      !method.name.startsWith("printf") &&
      !method.name.toLowerCase.contains("global") &&
      !method.name.toLowerCase.contains("operator")
    ) {

      method.dotCpg14.l match {
        case List(dotString: String) =>
          // 提取节点和边的定义
          val nodesAndEdges = extractNodesAndEdges(dotString)

          // 将新的节点和边添加到合并图中(避免重复)
          nodesAndEdges.split("\n").foreach { line =>
            if (line.trim.nonEmpty) {
              if (line.contains("->")) {
                if (!processedEdges.contains(line)) {
                  combinedDotString += line + "\n"
                  processedEdges += line
                }
              } else {
                if (!processedNodes.contains(line)) {
                  // 提取节点ID
                  val nodeIdPattern = """(\d+)""".r
                  val modifiedLine = nodeIdPattern.findFirstIn(line) match {
                    case Some(idStr) if blockingNodes.contains(idStr.toLong) =>
                      // 在节点定义中添加shape属性
                      line.replaceFirst(
                        """(\"\d+\") \[(label = .*)(\])""",
                        "$1 [shape=diamond,$2$3"
                      )
                    case _ => line
                  }
                  combinedDotString += modifiedLine + "\n"
                  processedNodes += line
                }
              }
            }
          }
      }
    }
  }

  // 添加pthread_create边
  cpg.call.name("pthread_create").foreach { createCall =>
    // 获取线程函数参数
    createCall.argument(3) match {
      case threadFunc: nodes.Expression =>
        // 获取目标函数
        val targetMethod = cpg.method.name(threadFunc.code).head
        // 添加create边
        combinedDotString += s"""  "${createCall.id}" -> "${targetMethod.id}" [color=red,label="creates thread"];\n"""
    }
  }

  // 添加pthread_join边
  cpg.call.name("pthread_join").foreach { joinCall =>
    // 获取当前函数的return节点
    val returnNode = joinCall.method.ast.isReturn.head
    // 添加join边
    combinedDotString += s"""  "${joinCall.id}" -> "${returnNode.id}" [color=blue,label="joins thread"];\n"""
  }

  combinedDotString += "}\n"

  // 保存合并后的图
  val outputPath = s"${outputDir}/combined_functions.dot"
  new java.io.PrintWriter(outputPath) { write(combinedDotString); close() }
  println(s"合并图已保存到: $outputPath")
}

// 辅助函数：从dot字符串中提取节点和边的定义
def extractNodesAndEdges(dotString: String): String = {
  val lines = dotString.split("\n")
  lines
    .filter(line =>
      !line.trim.startsWith("digraph") &&
        !line.trim.startsWith("}") &&
        line.trim.nonEmpty
    )
    .mkString("\n") + "\n"
}
