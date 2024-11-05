import re
from graphviz import Source
import pydot

def parse_dot_label(label_str):
    """解析节点的label属性
    
    Args:
        label_str: 形如 '<(METHOD,main)<SUB>12</SUB>>' 的标签字符串
        
    Returns:
        dict: 包含节点类型、名称、行号等信息的字典
    """
    # 移除HTML标签
    label_str = label_str.replace('<', '').replace('>', '')
    
    # 提取行号
    line_num = None
    sub_match = re.search(r'<SUB>(\d+)</SUB>', label_str)
    if sub_match:
        line_num = int(sub_match.group(1))
        label_str = re.sub(r'<SUB>\d+</SUB>', '', label_str)
    
    # 解析括号内的内容
    match = re.match(r'\(([\w_]+),([^)]+)\)', label_str)
    if match:
        node_type = match.group(1)
        content = match.group(2)
        
        return {
            'type': node_type,
            'content': content.strip(),
            'line': line_num
        }
    return None

def parse_dot_file(dot_file_path):
    """解析整个dot文件
    
    Args:
        dot_file_path: dot文件的路径
        
    Returns:
        dict: 包含所有节点和边的信息
    """
    # 读取dot文件
    graphs = pydot.graph_from_dot_file(dot_file_path)
    if not graphs:
        return None
    
    graph = graphs[0]
    
    # 解析结果
    result = {
        'nodes': {},
        'edges': []
    }
    
    # 处理节点
    for node in graph.get_nodes():
        node_id = node.get_name().strip('"')
        label = node.get('label')
        
        if label:
            # 解析label属性
            node_info = parse_dot_label(label)
            if node_info:
                result['nodes'][node_id] = node_info
    
    # 处理边
    for edge in graph.get_edges():
        source = edge.get_source().strip('"')
        dest = edge.get_destination().strip('"')
        edge_type = edge.get_label() or ''
        if edge_type:
            edge_type = edge_type.strip('"')
        
        result['edges'].append({
            'source': source,
            'target': dest,
            'type': edge_type
        })
    
    return result

def analyze_control_flow(parsed_data):
    """分析控制流
    
    Args:
        parsed_data: parse_dot_file的返回结果
    """
    # 找出所有控制结构节点
    control_nodes = {
        node_id: info 
        for node_id, info in parsed_data['nodes'].items() 
        if info['type'] == 'CONTROL_STRUCTURE'
    }
    
    # 找出所有方法节点
    method_nodes = {
        node_id: info 
        for node_id, info in parsed_data['nodes'].items() 
        if info['type'] == 'METHOD'
    }
    
    # 找出所有块节点
    block_nodes = {
        node_id: info 
        for node_id, info in parsed_data['nodes'].items() 
        if info['type'] == 'BLOCK'
    }
    
    return {
        'control_structures': control_nodes,
        'methods': method_nodes,
        'blocks': block_nodes
    }

def merge_graphs(main_data, thread_data, 
                 pthread_create_node, 
                 pthread_join_node, 
                 thread_method_node,
                 thread_return_node):
    """
    合并主图和线程函数图的数据
    
    Args:
        main_data: main.dot的解析数据
        thread_data: threadFunction.dot的解析数据
        pthread_create_node: pthread_create节点的ID
        pthread_join_node: join节点的ID
        thread_method_node: threadFunction方法起始节点的ID
        thread_return_node: threadFunction方法返回节点的ID
    """
    # 合并节点
    main_data['nodes'].update(thread_data['nodes'])
    
    # 合并边
    main_data['edges'].extend(thread_data['edges'])
    
    # 添加从pthread_create到threadFunction方法的边
    main_data['edges'].append({
        'source': pthread_create_node,
        'target': thread_method_node,
        'type': 'SYNC'
    })

    main_data['edges'].append({
        'source': thread_return_node,
        'target': pthread_join_node,
        'type': 'SYNC'
    })

def remove_isolated_nodes(graph):
    """删除没有入边和出边的孤立节点"""
    while True:
        isolated_nodes = []
        for node in graph.get_nodes():
            # 获取所有与该节点相关的边
            edges = graph.get_edges()
            has_connection = False
            
            # 检查节点是否有入边或出边
            for edge in edges:
                if edge.get_source() == node.get_name() or edge.get_destination() == node.get_name():
                    has_connection = True
                    break
            
            # 如果节点没有连接，添加到待删除列表
            if not has_connection:
                isolated_nodes.append(node)
        
        # 如果没有孤立节点，退出循环
        if not isolated_nodes:
            break
            
        # 删除孤立节点
        for node in isolated_nodes:
            graph.del_node(node.get_name())
            # print(f"删除孤立节点: {node.get_name()}")

# 使用示例
if __name__ == "__main__":
    dot_path = "./dot/"
    dot_file = "./dot/main.dot"
    
    # 解析dot文件
    main_data = parse_dot_file(dot_file)

    # 查找pthread_create节点和threadFunction引用
    pthread_create_node = None
    pthread_join_node = None
    thread_method_node = None
    thread_return_node = None

    main_method_name = None
    thread_method_name = None

    for node_id, info in main_data['nodes'].items():
        if info['type'] == 'CONTROL_STRUCTURE':
            # print(f"Found control structure: {info['content']} at line {info['line']}")
            if 'pthread_create' in info['content']:
                pthread_create_node = node_id
                pattern = r"pthread_create\([^)]*?,\s*[^,]*?,\s*([^,]*?),"
                match = re.search(pattern, info['content'])
                if match:
                # 输出第二个逗号到第三个逗号之间的内容
                    thread_method_name = match.group(1)
                    print(thread_method_name)
                else:
                    print("No match found")
                # print(pthread_create_node)
            elif 'pthread_join' in info['content']:
                pthread_join_node = node_id
                # print(pthread_join_node)


    if pthread_create_node and thread_method_name:
        thread_dot_path = dot_path + thread_method_name+ ".dot"
        try:
            thread_data = parse_dot_file(thread_dot_path)
            
            for node_id, info in thread_data['nodes'].items():
                if info['type'] == 'METHOD':
                    thread_method_node = node_id
                elif info['type'] == 'METHOD_RETURN':
                    thread_return_node = node_id
            
            if pthread_join_node and thread_return_node:
                # 合并图
                merge_graphs(main_data, thread_data, pthread_create_node, pthread_join_node, thread_method_node, thread_return_node)
                print("成功合并main.dot和threadFunction.dot")
                graph = pydot.Dot(graph_type='digraph')
                # 添加所有节点
                for node_id, node_info in main_data['nodes'].items():
                    # 构建label字符串
                    label = f"({node_info['type']},{node_info['content']})"
                    if node_info['line']:
                        label = f"{label}<SUB>{node_info['line']}</SUB>"
                    
                    # 创建节点
                    node = pydot.Node(node_id, label=label)
                    graph.add_node(node)
                
                # 添加所有边
                for edge in main_data['edges']:
                    # 确保源节点和目标节点都存在
                    if edge['source'] in main_data['nodes'] and edge['target'] in main_data['nodes']:
                        # 创建边
                        new_edge = pydot.Edge(
                            str(edge['source']),  # 确保转换为字符串
                            str(edge['target']),  # 确保转换为字符串
                            label=str(edge['type']) if edge['type'] else ""  # 确保标签也是字符串
                        )
                        graph.add_edge(new_edge)
                        # print(f"跳过无效边: {edge['source']} -> {edge['target']}")

                remove_isolated_nodes(graph)
                # 保存到文件
                new_graph_name = dot_path + "new.dot"
                graph.write_raw(new_graph_name)
        except FileNotFoundError:
            print(f"未找到{thread_dot_path}文件")
    else:
        print("在main.dot中未找到pthread_create节点或threadFunction引用")
    

    # if main_data:
    #     # 打印所有METHOD节点
    #     for node_id, info in main_data['nodes'].items():
    #         if info['type'] == 'METHOD_RETURN':
    #             print(f"Found method: {info['content']} at line {info['line']}")
        
    #     # 打印所有控制结构
    #     for node_id, info in parsed_data['nodes'].items():
    #         if info['type'] == 'CONTROL_STRUCTURE':
    #             print(f"Found control structure: {info['content']} at line {info['line']}")
        
    #     # 打印所有BLOCK节点
    #     for node_id, info in parsed_data['nodes'].items():
    #         if info['type'] == 'BLOCK':
    #             print(f"Found block: {info['content']} at line {info['line']}")
        
    #     # 分析控制流
    #     control_flow = analyze_control_flow(parsed_data)
    #     print("\nControl flow analysis:")
    #     print(f"Found {len(control_flow['methods'])} methods")
    #     print(f"Found {len(control_flow['control_structures'])} control structures")
    #     print(f"Found {len(control_flow['blocks'])} blocks")