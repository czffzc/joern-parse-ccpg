package joern$minusexport$minusdemo


final class extract_func$_ {
def args = extract_func_sc.args$
def scriptPath = """joern-export-demo/extract_func.sc"""
/*<script>*/
@main def exec(codeDir: String, outputDir: String) = {
  // 使用默认路径
  val defaultCodeDir = "/home/kevin/joern-parse/joern-export-demo/c_code"
  val defaultOutputDir = "/home/kevin/joern-parse/joern-export-demo/dot/"

  println(s"Import Code Path: $codeDir")
  println(s"Output Directory: $outputDir")

  // 从命令行参数获取输出目录
  // val outputDir = if (args.length > 0) args(6) else {
  //   // 默认路径
  //   "/home/kevin/joern-parse/joern-export-demo/dot/"
  // }
  // val outputDir = "/home/kevin/joern-parse/joern-export-demo/dot/"
  new java.io.File(outputDir).mkdirs()
  importCode(codeDir)
  markLockProtectedNodes()
  // 导出所有函数的CFG
  filterMethod(outputDir)
  createCombinedThreadGraph(outputDir)
}

def createCombinedThreadGraph(outputDir: String) = {
  println("\n创建组合调用图:")
  println("====================")
// 存储所有需要合并的图
  var combinedDotString = "digraph combined {\n"
  var callThreadName = ""
  cpg.method.name("main").foreach { mainMethod =>
    // 获取main函数的图
    val mainGraph = mainMethod.dotCpg14.l.headOption.getOrElse("")

    // 遍历main中的pthread_create调用
    mainMethod.call.name("pthread_create").foreach { pthreadCall =>
      val threadFunction = pthreadCall.argument(3)

      // 获取线程函数名
      callThreadName = threadFunction match {
        case identifier: nodes.Identifier => identifier.name
        case call: nodes.Call             => call.name
        case _                            => threadFunction.code
      }

      // 查找对应的线程函数
      cpg.method.name(callThreadName).foreach { threadMethod =>
        // 获取线程函数的图
        val threadGraph = threadMethod.dotCpg14.l.headOption.getOrElse("")

        // 提取两个图的节点和边（去掉digraph声明部分）
        val mainNodes = extractNodesAndEdges(mainGraph)
        val threadNodes = extractNodesAndEdges(threadGraph)

        // 获取pthread_create调用节点的ID和线程函数入口节点的ID
        val callNodeId = pthreadCall.id
        val threadEntryId = threadMethod.id

        // 合并节点和边
        combinedDotString += mainNodes
        combinedDotString += threadNodes

        // 添加调用点到线程函数入口的边
        combinedDotString += s"""  "${callNodeId}" -> "${threadEntryId}" [color=red,label="creates thread"];\n"""

      }
      // 遍历pthread_join调用
      mainMethod.call.name("pthread_join").foreach { joinCall =>
        println(s"找到pthread_join调用: ${joinCall.code}")

        // 查找对应的线程函数（使用之前保存的threadFuncName）
        cpg.method.name(callThreadName).foreach { threadMethod =>
          // 获取线程函数的所有return节点
          threadMethod.ast.isReturn.foreach { returnNode =>
            // 添加pthread_join边到return节点
            combinedDotString += s"""  "${joinCall.id}" -> "${returnNode.id}" [color=blue,label="joins thread"];\n"""
          }
        }
      }
      combinedDotString += "}\n"

      // 保存合并后的图
      val fileName = s"${mainMethod.name}_to_${callThreadName}_thread.dot"
      val outputPath = s"${outputDir}/$fileName"
      new java.io.PrintWriter(outputPath) { write(combinedDotString); close() }
      println(s"合并图已保存到: $outputPath")
    }
  }
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

def filterMethod(outputDir: String) = {
  cpg.method.foreach { method =>
    // 过滤掉不需要的函数
    if (
      !method.name.startsWith("pthread") &&
      !method.name.startsWith("perror") &&
      !method.name.startsWith("printf") &&
      !method.name.toLowerCase.contains("global") && // 不区分大小写检查global
      !method.name.toLowerCase.contains("operator")
    ) { // 不区分大小写检查operator

      val methodName =
        method.name.replaceAll("[^a-zA-Z0-9]", "_") // 清理文件名，避免非法字符

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

def markLockProtectedNodes() = {
  println("\n分析锁保护区域:")
  println("====================")

  cpg.method.foreach { method =>
    // 找到所有的lock和unlock调用
    val lockNodes = method.call.name(".*lock").l
    val unlockNodes = method.call.name(".*unlock").l

    // 存储需要标记的节点
    var protectedNodes = Set[nodes.AstNode]()

    lockNodes.foreach { lockNode =>
      unlockNodes.foreach { unlockNode =>
        val nodesInBetween = lockNode.ast
          .takeWhile(node => node.id != unlockNode.id)
          .filter(node => node != lockNode && node != unlockNode)
          .l

        protectedNodes ++= nodesInBetween
      }
    }

    // 标记保护区域内的节点
    protectedNodes.foreach { node =>
      node.property("color", "green")
    }
  }
}

/*</script>*/ /*<generated>*//*</generated>*/
}

object extract_func_sc {
  private var args$opt0 = Option.empty[Array[String]]
  def args$set(args: Array[String]): Unit = {
    args$opt0 = Some(args)
  }
  def args$opt: Option[Array[String]] = args$opt0
  def args$: Array[String] = args$opt.getOrElse {
    sys.error("No arguments passed to this script")
  }

  lazy val script = new extract_func$_

  def main(args: Array[String]): Unit = {
    args$set(args)
    val _ = script.hashCode() // hashCode to clear scalac warning about pure expression in statement position
  }
}

export extract_func_sc.script as `extract_func`

