import pygraphviz as pgv
import pandas as pd
import re


def parse_label(label):
    """
    Parse a given label to extract relevant attributes.
    """
    attributes = {}
    # Regular expression to match attribute key-value pairs
    matches = re.findall(r'(\w+)=["\']([^"\']+)["\']', label)
    for key, value in matches:
        attributes[key] = value
    return attributes

def read_dot_file(file_path):
    """
    Read a .dot file and parse the nodes.
    """
    # 检查文件是否存在
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
      
    # Load the .dot file using pygraphviz
    graph = pgv.AGraph(file_path)
    
    # Prepare a list to store node information
    nodes_data = []
    
    # Iterate through each node in the graph
    for node in graph.nodes():
        node_id = node.get_name()
        label = node.attr.get('label', 'No label')
        code = node.attr.get('code', 'No code')
        print(f'Node ID: {node_id}, Label: {label}, Code: {code}')
        # label = node.attr['label']
        
        # # Parse the label to extract detailed attributes
        # parsed_attributes = parse_label(label)
        # parsed_attributes['node_id'] = node_id  # Add the node ID
        
        # nodes_data.append(parsed_attributes)
    
    # Convert to a pandas DataFrame for better readability
    return pd.DataFrame(nodes_data)

def extract_custom_properties(file_path, properties_to_extract=None):
    """
    Extract specified properties from a custom dot file using pygraphviz.

    :param file_path: Path to the dot file
    :param properties_to_extract: List of properties to extract (e.g., ["label", "CODE", "TYPE_FULL_NAME"])
    :return: List of dictionaries containing extracted properties
    """
    if properties_to_extract is None:
        # If no specific properties are given, extract all found properties
        properties_to_extract = ["label", "CODE", "TYPE_FULL_NAME", "NAME", "ORDER"]

    # Load the dot file with pygraphviz
    graph = pgv.AGraph(file_path)

    # Initialize a list to store extracted properties
    extracted_data = []

    # Define a regex pattern to find key-value pairs within the label attribute
    key_value_pattern = re.compile(r'(\w+)=[.*?]')

    # Iterate over all nodes and extract specified properties
    for node in graph.nodes():
        node_data = {}
        node_id = node.get_name()
        node_data['node_id'] = node_id

        # Get the label content if it exists
        label = node.attr.get('label', '')

        # Use regex to find all key-value pairs in the label
        label_properties = key_value_pattern.findall(label)

        # Convert the key-value pairs into a dictionary
        label_dict = {key: value for key, value in label_properties}

        # Filter the properties to only keep the ones specified in `properties_to_extract`
        filtered_properties = {prop: label_dict.get(prop, '') for prop in properties_to_extract}
        filtered_properties['node_id'] = node_id  # Add the node_id

        extracted_data.append(filtered_properties)

    return extracted_data


file_path = 'export_escaped.dot'
properties = ["label", "CODE", "TYPE_FULL_NAME", "NAME", "ORDER"]  # Specify the properties you want to extract
extracted_data = extract_custom_properties(file_path, properties)

# Display the extracted data
import pandas as pd
df = pd.DataFrame(extracted_data)
print(df)

# with open(file_path, 'r') as f:
#     print("DOT文件内容前10行:")
#     for i, line in enumerate(f):
#         if i < 10:
#             print(f"{i+1}: {line.strip()}")
#         else:
#             break

# nodes_df = read_dot_file(file_path)

# Display the DataFrame to the user
# import ace_tools as tools; 
# tools.display_dataframe_to_user(name="Parsed CPG Graph Nodes", dataframe=nodes_df)
# Display the DataFrame
# print(nodes_df)