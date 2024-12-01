@main def main(outputDir: String) = {
    importCode("/home/kevin/joern-parse/joern-parse-demo/c_code")
    // 创建输出目录
    new java.io.File(outputDir).mkdirs()
    
    // 获取所有节点和边
    val nodes = cpg.all.l
    val edges = cpg.edges.l  // 直接从cpg获取edges
    
    // 构建dot格式的内容
    val nodesDot = nodes.map { node =>
        s""""${node.id}" [label="${node.label}\\n${escapeString(node.toString)}"];"""
    }
    
    val edgesDot = edges.map { edge =>
        s""""${edge.inNode.id}" -> "${edge.outNode.id}" [label="${edge.label}"];"""
    }
    
    // 合并所有内容并写入文件
    val dot = (nodesDot ++ edgesDot).distinct.mkString(
        "digraph G {\n",
        "\n",
        "\n}"
    )
    
    os.write(os.Path(s"$outputDir/graph.dot"), dot)
}

// 辅助函数:转义字符串中的特殊字符
def escapeString(str: String): String = {
    str.replace("\"", "\\\"")
       .replace("\n", "\\n")
       .take(50) // 限制标签长度
}