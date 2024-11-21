@main def main(codeDir: String, outputDir: String) = {
  importCode(codeDir)
  java.io.File(outputDir).mkdirs()
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

  // 存储需要标记的节点ID和边
  var blockingNodes = Set[Long]()
  var blockingEdges = Set[String]()

  // 标记所有lock-unlock之间的节点和边
  cpg.method.foreach { method =>
    val lockNodes = method.cfgNode.isCall.name(".*lock").l

    lockNodes.foreach { lockNode =>
      // 对于每个lock节点，找到最近的unlock节点及其路径
      val pathsToUnlock = lockNode.cfgNext
        .repeat(_.cfgNext)(_.until(_.isCall.name(".*unlock")))
        .enablePathTracking
        .path
        .l

      pathsToUnlock.foreach { path =>
        // 将path转换为StoredNode列表以访问id属性
        val nodesInPath = path.collect { case node: StoredNode =>
          node
        }.toList

        // 添加节点到阻塞集合
        blockingNodes =
          blockingNodes ++ Set(lockNode.id()) ++ nodesInPath.map(_.id())

        // 处理路径上的边
        nodesInPath.sliding(2).foreach {
          case List(source, target) =>
            // 构建边的字符串表示
            val edgeStr = s"""  "${source.id()}" -> "${target.id()}" """
            val edgePattern = s"""$edgeStr  \\[ label = "(.*?)" \\]""".r

            // 检查边是否已存在，如果存在则添加color属性
            val newEdgeStr =
              combinedDotString.split("\n").find(_.startsWith(edgeStr)) match {
                case Some(existingEdge) =>
                  // 提取现有的label
                  edgePattern.findFirstMatchIn(existingEdge) match {
                    case Some(m) =>
                      val label = m.group(1)
                      s"""$edgeStr  [color=red,label = "$label"]"""
                    case None =>
                      s"""$edgeStr  [color=red]"""
                  }
                case None =>
                  s"""$edgeStr  [color=red]"""
              }

            if (!processedEdges.contains(newEdgeStr)) {
              combinedDotString += newEdgeStr + "\n"
              processedEdges += newEdgeStr
            }
          case _ => // 忽略其他情况
        }
      }
    }
  }

  // 标记pthread_join相关的节点
  cpg.call.name("pthread_join").foreach { joinCall =>
    blockingNodes += joinCall.id
  }

  // 处理每个函数
  cpg.method.foreach { method =>
    method.dotCpg14.l match {
      case List(dotString: String) =>
        // 提取节点和边的定义
        val nodesAndEdges = extractNodesAndEdges(dotString)

        // 处理函数调用和返回边
        method.call.foreach { callNode =>
          // 获取被调用的方法
          callNode.callee.foreach { callee =>
            // 构造从被调用函数返回到调用点的边
            val returnEdgeStr = s"""  "${callee.id()}" -> "${callNode
                .id()}"  [color=blue,style=dashed,label = "return"]"""
            if (!processedEdges.contains(returnEdgeStr)) {
              combinedDotString += returnEdgeStr + "\n"
              processedEdges += returnEdgeStr
            }
          }
        }
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

      case _ =>
    }
  }

  // 添加pthread_create边
  cpg.call.name("pthread_create").foreach { createCall =>
    try {
      createCall.argument(3) match {
        case threadFunc: nodes.Expression =>
          val targetMethods = cpg.method.name(threadFunc.code).l
          targetMethods match {
            case methods if methods.nonEmpty =>
              println(
                s"找到${methods.size}个匹配函数: ${methods.map(_.name).mkString(", ")}"
              )
              // 为每个匹配的方法创建一个连接
              methods.foreach { method =>
                combinedDotString += s"""  "${createCall.id}" -> "${method.id}" [color=red,label="creates thread"];\n"""
              }
            case Nil =>
              println(s"警告: 未找到匹配的线程函数: ${threadFunc.code}")
          }
        case _ =>
          println(s"警告: pthread_create的第3个参数不是预期的表达式类型")
      }
    } catch {
      case e: Exception =>
        println(s"处理pthread_create时发生错误: ${e.getMessage}")
        e.printStackTrace()
    }
  }

  // 添加pthread_join边
  cpg.call.name("pthread_join").foreach { joinCall =>
    try {
      // 获取调用pthread_join的函数
      val callingMethod = joinCall.method

      // 获取调用函数中的所有return节点
      val returnNodes = callingMethod.ast.isReturn.l
      if (returnNodes.nonEmpty) {
        // 如果有return节点,添加join边到第一个return节点
        combinedDotString += s"""  "${joinCall.id}" -> "${returnNodes.head.id}" [color=blue,label="joins thread"];\n"""
      } else {
        // 如果没有显式的return节点,尝试找到函数的最后一个语句或块
        val lastNode = callingMethod.ast.l.lastOption
        lastNode match {
          case Some(node) =>
            combinedDotString += s"""  "${joinCall.id}" -> "${node.id}" [color=blue,label="joins thread"];\n"""
          case None =>
            println(s"警告: 在函数 ${callingMethod.name} 中未找到合适的目标节点")
        }
      }
    } catch {
      case e: Exception =>
        println(s"处理pthread_join时发生错误: ${e.getMessage}")
        e.printStackTrace()
    }
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
        !(line.trim == "}") &&
        line.trim.nonEmpty
    )
    .mkString("\n") + "\n"
}
