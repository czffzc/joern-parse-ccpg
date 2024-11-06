# 通过joern解析cpg.bin文件获得各个函数的cpg
output_dir = "/path/to"
cmd = f'joern --script extract_func.sc --params "{output_dir} --"'
