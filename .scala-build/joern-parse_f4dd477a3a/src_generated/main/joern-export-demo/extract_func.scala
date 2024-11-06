package joern$minusexport$minusdemo


final class extract_func$_ {
def args = extract_func_sc.args$
def scriptPath = """joern-export-demo/extract_func.sc"""
/*<script>*/
@main def main(codeDir: String, outputDir: String) = {
  // 使用默认路径
  val defaultCodeDir = "/home/kevin/joern-parse/joern-export-demo/c_code"
  val defaultOutputDir = "/home/kevin/joern-parse/joern-export-demo/dot/"
 
  println(s"Import Code Path: $codeDir")
  println(s"Output Directory: $outputDir")
 

// importCode("/home/kevin/joern-parse/joern-export-demo/c_code")

// // 从命令行参数获取输出目录
// // val outputDir = if (args.length > 0) args(6) else {
// //   // 默认路径
// //   "/home/kevin/joern-parse/joern-export-demo/dot/"
// // }
// // val outputDir = "/home/kevin/joern-parse/joern-export-demo/dot/"
// new java.io.File(outputDir).mkdirs()

// // 遍历每个函数
// cpg.method.foreach { method =>
//   // 过滤掉不需要的函数
//   if (!method.name.startsWith("pthread") && 
//       !method.name.startsWith("perror") && 
//       !method.name.startsWith("printf") &&
//       !method.name.toLowerCase.contains("global") &&  // 不区分大小写检查global
//       !method.name.toLowerCase.contains("operator")) {  // 不区分大小写检查operator
    
//     val methodName = method.name.replaceAll("[^a-zA-Z0-9]", "_")  // 清理文件名，避免非法字符

//     val dotFilePath = outputDir + methodName + ".dot"
//     method.dotCpg14.l match {
//       case List(dotString: String) =>
//             val writer = new java.io.PrintWriter(dotFilePath)
//             writer.write(dotString)
//             writer.close()
//             println(s"Exported CPG for function $methodName to $dotFilePath")
//           case _ =>
//             println(s"Failed to export CPG for function $methodName")
//         }
//     }
// }
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

