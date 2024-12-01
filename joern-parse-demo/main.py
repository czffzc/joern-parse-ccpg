import re
import os
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

class Node:
    def __init__(self, id, label_dict):
        self.id = id
        self.attributes = label_dict
    
    def __str__(self):
        return f"Node {self.id}: {self.attributes}"

def parse_cpg_dot(file_path):
    """解析CPG dot文件并提取节点信息"""
    
    # 存储所有节点的字典
    nodes = {}
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取节点定义
    # 匹配格式: number[label=TYPE attr1="value1" attr2="value2" ...]
    node_pattern = r'(\d+)\[label=(\w+)\s([^\]]+)\]'
    
    # 查找所有节点定义
    for match in re.finditer(node_pattern, content):
        node_id = match.group(1)
        node_type = match.group(2)
        attributes_str = match.group(3)
        
        # 解析属性
        # 匹配格式: key="value"
        attr_pattern = r'(\w+)="([^"]*)"'
        attributes = {
            'TYPE': node_type,
            **dict(re.findall(attr_pattern, attributes_str))
        }
        
        # 创建节点对象并存储
        nodes[node_id] = Node(node_id, attributes)
    
    return nodes

def print_node_stats(nodes):
    """打印节点统计信息"""
    print(f"总节点数: {len(nodes)}")
    
    # 统计不同类型的节点
    type_count = {}
    for node in nodes.values():
        node_type = node.attributes['TYPE']
        type_count[node_type] = type_count.get(node_type, 0) + 1
    
    print("\n节点类型统计:")
    for node_type, count in type_count.items():
        print(f"{node_type}: {count}")

def find_concurrent_nodes(nodes):
    """识别并发相关的节点"""
    concurrent_apis = {
        'pthread_create',
        'pthread_join',
        'pthread_mutex_lock',
        'pthread_mutex_unlock',
        'pthread_mutex_init',
        'pthread_mutex_destroy',
        'pthread_exit',
        'pthread_cond_wait',
        'pthread_cond_signal',
        'sem_init',
        'sem_wait',
        'sem_post'
    }
    
    concurrent_nodes = {}
    for node_id, node in nodes.items():
        # 检查CALL节点的方法名
        if node.attributes.get('TYPE') == 'CALL':
            method_name = node.attributes.get('METHOD_FULL_NAME', '')
            if any(api in method_name for api in concurrent_apis):
                concurrent_nodes[node_id] = node
                
    return concurrent_nodes

def build_node_graph(file_path):
    """构建节点关系图"""
    edges = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配边的定义: node1 -> node2 [label=TYPE]
    edge_pattern = r'(\d+)\s*->\s*(\d+)\s*\[label=([^\]]+)\]'
    
    for match in re.finditer(edge_pattern, content):
        from_node = match.group(1)
        to_node = match.group(2)
        edge_type = match.group(3)
        
        if from_node not in edges:
            edges[from_node] = []
        edges[from_node].append((to_node, edge_type))
    
    return edges

def slice_concurrent_operations(nodes, edges, concurrent_nodes):
    """对并发操作进行多维度切片分析"""
    
    def backward_slice(node_id, visited=None, max_depth=10):
        """向后切片，追踪数据依赖和控制依赖"""
        if visited is None:
            visited = set()
        
        if node_id in visited or len(visited) > max_depth:
            return set()
        
        visited.add(node_id)
        slice_nodes = {node_id}
        
        # 遍历所有指向当前节点的边
        for from_node, edge_list in edges.items():
            for to_node, edge_type in edge_list:
                if to_node == node_id:
                    # 考虑更多类型的依赖关系
                    if any(dep in edge_type for dep in [
                        'REACHING_DEF',
                        'CONTROL_STRUCTURE',
                        'CFG',
                        'CDG',
                        'DDG',
                        'EVAL_TYPE',
                        'PARAMETER_LINK'
                    ]):
                        slice_nodes.update(backward_slice(from_node, visited.copy(), max_depth))
        
        return slice_nodes

    def forward_slice(node_id, visited=None, max_depth=10):
        """向前切片，追踪数据影响"""
        if visited is None:
            visited = set()
        
        if node_id in visited or len(visited) > max_depth:
            return set()
        
        visited.add(node_id)
        slice_nodes = {node_id}
        
        # 遍历从当前节点出发的边
        if node_id in edges:
            for to_node, edge_type in edges[node_id]:
                if any(dep in edge_type for dep in [
                    'REACHING_DEF',
                    'CONTROL_STRUCTURE',
                    'CFG',
                    'CDG',
                    'DDG'
                ]):
                    slice_nodes.update(forward_slice(to_node, visited.copy(), max_depth))
        
        return slice_nodes

    def context_slice(node_id, context_size=5):
        """提取节点周围的上下文切片"""
        slice_nodes = set()
        node = nodes.get(node_id)
        if not node or 'LINE_NUMBER' not in node.attributes:
            return slice_nodes
        
        current_line = int(node.attributes['LINE_NUMBER'])
        # 获取指定行数范围内的所有节点
        for other_id, other_node in nodes.items():
            if 'LINE_NUMBER' in other_node.attributes:
                other_line = int(other_node.attributes['LINE_NUMBER'])
                if abs(other_line - current_line) <= context_size:
                    slice_nodes.add(other_id)
        
        return slice_nodes

    def get_function_slice(node_id):
        """提取整个函数的切片"""
        slice_nodes = set()
        node = nodes.get(node_id)
        if not node:
            return slice_nodes
        
        # 找到函数定义节点
        function_name = None
        if 'METHOD_FULL_NAME' in node.attributes:
            function_name = node.attributes['METHOD_FULL_NAME'].split('(')[0]
        
        if function_name:
            for other_id, other_node in nodes.items():
                if ('METHOD_FULL_NAME' in other_node.attributes and 
                    other_node.attributes['METHOD_FULL_NAME'].startswith(function_name)):
                    slice_nodes.add(other_id)
        
        return slice_nodes

    # 生成多维度切片
    slices = {
        'backward': {},
        'forward': {},
        'context': {},
        'function': {},
        'combined': {}
    }
    
    for node_id in concurrent_nodes:
        # 生成各种类型的切片
        backward_nodes = backward_slice(node_id)
        forward_nodes = forward_slice(node_id)
        context_nodes = context_slice(node_id)
        function_nodes = get_function_slice(node_id)
        
        # 存储各类切片
        slices['backward'][node_id] = backward_nodes
        slices['forward'][node_id] = forward_nodes
        slices['context'][node_id] = context_nodes
        slices['function'][node_id] = function_nodes
        
        # 合并切片
        combined_nodes = backward_nodes | forward_nodes | context_nodes
        slices['combined'][node_id] = combined_nodes
    
    return slices

def extract_slice_code(nodes, slice_nodes):
    """从切片中提取代码"""
    # 按行号排序的代码片段
    code_lines = {}
    
    for node_id in slice_nodes:
        node = nodes.get(node_id)
        if node and 'LINE_NUMBER' in node.attributes and 'CODE' in node.attributes:
            line_num = int(node.attributes['LINE_NUMBER'])
            code = node.attributes['CODE'].strip()
            if code and code != '<empty>':
                code_lines[line_num] = code
    
    # 按行号排序并组合代码
    sorted_lines = []
    for line_num in sorted(code_lines.keys()):
        sorted_lines.append(f"{code_lines[line_num]}")
    
    return '\n'.join(sorted_lines)

def extract_slice_subgraph(nodes, edges, slice_nodes):
    """从切片中提取子图"""
    subgraph = "digraph {\n"
    
    # 添加节点
    for node_id in slice_nodes:
        if node_id in nodes:
            node = nodes[node_id]
            # 重建节点定义字符串
            attrs = []
            for key, value in node.attributes.items():
                if key != 'TYPE':  # TYPE已经在label中
                    attrs.append(f'{key}="{value}"')
            node_str = f'  {node_id}[label={node.attributes["TYPE"]} {" ".join(attrs)}]\n'
            subgraph += node_str
    
    # 添加边
    for from_node in edges:
        if from_node in slice_nodes:  # 只处理切片中的节点
            for to_node, edge_type in edges[from_node]:
                if to_node in slice_nodes:  # 确保目标节点也在切片中
                    edge_str = f'  {from_node} -> {to_node} [label={edge_type}]\n'
                    subgraph += edge_str
    
    subgraph += "}"
    return subgraph

def process_node_slice(nodes, edges, node_id, concurrent_nodes, slice_type):
    """处理单个节点的切片生成"""
    try:
        slices = slice_concurrent_operations(nodes, edges, {node_id: concurrent_nodes[node_id]})
        node = nodes[node_id]
        method_name = node.attributes.get('METHOD_FULL_NAME', 'unknown')
        
        for slice_type, type_slices in slices.items():
            for _, slice_nodes in type_slices.items():
                # 提取切片代码
                # code = extract_slice_code(nodes, slice_nodes)
                # if code:
                #     filename = f"slices/{slice_type}_{node_id}_{method_name.replace('/', '_')}.txt"
                #     os.makedirs(os.path.dirname(filename), exist_ok=True)
                #     with open(filename, 'w', encoding='utf-8') as f:
                #         f.write(f"// 切片类型: {slice_type}\n")
                #         f.write(f"// 源节点: {method_name}\n")
                #         f.write(f"// 相关节点数: {len(slice_nodes)}\n\n")
                #         f.write(code)
                    
                #     print(f"已保存切片到: {filename}")
                # 生成子图
                subgraph = extract_slice_subgraph(nodes, edges, slice_nodes)
                if subgraph:
                    filename = f"dot_slices/{slice_type}_{node_id}_{method_name.replace('/', '_')}.dot"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(subgraph)
                    
                    print(f"已保存切片图到: {filename}")
    except Exception as e:
        print(f"处理节点 {node_id} 时出错: {e}")

def main():
    file_path = "/home/kevin/joern-parse/joern-parse-demo/out/export.dot"
    
    try:
        # parse cpg dot file
        nodes = parse_cpg_dot(file_path)
        concurrent_nodes = find_concurrent_nodes(nodes)
        edges = build_node_graph(file_path)
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for node_id in concurrent_nodes:
                futures.append(
                    executor.submit(
                        process_node_slice, 
                        nodes, 
                        edges, 
                        node_id, 
                        concurrent_nodes,
                        'combined'
                    )
                )
            
            for future in futures:
                future.result()
        
    except Exception as e:
        print(f"分析出错: {e}")

if __name__ == "__main__":
    main()