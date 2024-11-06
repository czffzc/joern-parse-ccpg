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
  // new java.io.File(outputDir).mkdirs()

  findPthreadCreateCalls()
}

def findPthreadCreateCalls() = {
  println("\n分析pthread_create调用:")
  println("====================")

  importCode("/home/kevin/joern-parse/joern-export-demo/c_code")

  cpg
    .call("pthread_create")
    .foreach { call =>
      // 获取所在的函数
      val parentMethod = call.method.name
      val location = call.location

      // 获取pthread_create的第3个参数(函数指针)
      val threadFunction = call.argument(3)

      println(s"\n在函数 '$parentMethod' 中发现pthread_create调用:")
      println(s"位置: ${location.filename}:${location.lineNumber}")

      // 分析线程函数
      threadFunction match {
        case ref: nodes.Identifier =>
          println(s"线程函数名: ${ref.name}")

          // 查找线程函数的定义
          cpg
            .method(ref.name)
            .foreach { threadMethod =>
              println("\n线程函数详情:")
              println(
                s"- 定义位置: ${threadMethod.location.filename}:${threadMethod.location.lineNumber}"
              )
              println(s"- 参数个数: ${threadMethod.parameter.size}")
              println(s"- 函数签名: ${threadMethod.signature}")

              // 分析线程函数内的函数调用
              println("\n线程函数中的函数调用:")
              threadMethod.call
                .foreach { innerCall =>
                  println(
                    s"- ${innerCall.code} (行号: ${innerCall.location.lineNumber})"
                  )
                }
            }

        case call: nodes.Call =>
          println(s"线程函数通过函数调用获得: ${call.name}")
          println(s"函数调用代码: ${call.code}")

        case _ =>
          println(s"线程函数参数类型: ${threadFunction.getClass.getSimpleName}")
      }

      println(s"\n完整调用代码: ${call.code}")
      println("----------------------------------------")
    }

  if (cpg.call("pthread_create").isEmpty) {
    println("未发现pthread_create的调用")
  }
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

