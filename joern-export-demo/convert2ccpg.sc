@main def main(codeDir: String, outputDir: String) = {
  importCode(codeDir)
  java.io.File(outputDir).mkdirs()
  exportAllFunctionsToDot(outputDir)
}

def exportAllFunctionsToDot(outputDir: String) = {
  println("\n导出所有函数的合并CPG:")
  println("====================")

  var combinedDotString = "digraph combined_cpg {\n"
  var processedNodes = Set[String]()
  var processedEdges = Set[String]()

  cpg.method.foreach { method =>
    // 直接构建包含所需属性的节点定义
    method.ast.foreach { node =>
      val nodeId = node.id.toString
      if (!processedNodes.contains(nodeId)) {
        // 构建基本label
        val baseLabel = node.label
        val code = Option(node.code).getOrElse("")
        val lineNum = node.lineNumber.getOrElse("")
        val colNum = node.columnNumber.getOrElse("")
        
        // 构建完整的节点定义
        val nodeStr = s""""$nodeId" [label = <($baseLabel,$code)<SUB>$lineNum</SUB>> COLUMN_NUMBER="$colNum" LINE_NUMBER="$lineNum"]\n"""
        combinedDotString += nodeStr
        processedNodes += nodeId
      }
    }

    // 添加边
    method.ast.foreach { node =>
      // AST edges
      node._astOut.foreach { child =>
        // 提取纯数字ID
        val nodeId = node.id.toString.split("=").last.dropRight(2)  // 移除最后的 "].id"
        val childId = child.id.toString.split("=").last.dropRight(2)
        val edgeStr = s"""  "$nodeId" -> "$childId"  [ label = "AST: "] """
        if (!processedEdges.contains(edgeStr)) {
          combinedDotString += edgeStr + "\n"
          processedEdges += edgeStr
        }
      }

      // CFG edges
      node._cfgOut.foreach { succ =>
        val nodeId = node.id.toString.split("=").last.dropRight(2)
        val succId = succ.id.toString.split("=").last.dropRight(2)
        val edgeStr = s"""  "$nodeId" -> "$succId"  [ label = "CFG: "] """
        if (!processedEdges.contains(edgeStr)) {
          combinedDotString += edgeStr + "\n"
          processedEdges += edgeStr
        }
      }

      // CDG edges
      node._cdgOut.foreach { dependent =>
        val nodeId = node.id.toString.split("=").last.dropRight(2)
        val depId = dependent.id.toString.split("=").last.dropRight(2)
        val edgeStr = s"""  "$nodeId" -> "$depId"  [ label = "CDG: "] """
        if (!processedEdges.contains(edgeStr)) {
          combinedDotString += edgeStr + "\n"
          processedEdges += edgeStr
        }
      }
    }

    // 处理函数调用和返回边
    method.call.foreach { callNode =>
      callNode.callee.foreach { callee =>
        val returnEdgeStr = s"""  "${callee.id()}" -> "${callNode.id()}"  [color=blue,style=dashed,label = "return"]"""
        if (!processedEdges.contains(returnEdgeStr)) {
          combinedDotString += returnEdgeStr + "\n"
          processedEdges += returnEdgeStr
        }
      }
    }
  }

  // 添加pthread相关的边
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

  val outputPath = s"${outputDir}/combined_functions.dot"
  new java.io.PrintWriter(outputPath) { write(combinedDotString); close() }
  println(s"合并图已保存到: $outputPath")
}
