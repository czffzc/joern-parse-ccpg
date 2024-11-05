import re

def escape_nested_quotes_in_file(file_path):
    """
    Read a .dot file and escape all nested quotes in the content.
    """
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Define a function to escape quotes within CODE="..." content
    def escape_nested_quotes(match):
        # Extract the content between CODE=" and "COLUMN
        inner_content = match.group(1)
        # Escape all quotes within this segment
        escaped_content = inner_content.replace('"', '\\"')
        return f'CODE="{escaped_content}" COLUMN_NUMBER='

    # Use regex to find and replace nested quotes in CODE="..." segments within square brackets []
    # The regex uses DOTALL to allow for multi-line matching and ensures we're working within brackets
    escaped_content = re.sub(r'CODE="(.*?)"\s', 
                             lambda m: m.group(0).replace(m.group(1), m.group(1).replace('"', '\\"')), 
                             content, flags=re.DOTALL)

    # Save the escaped content back to the file or to a new file
    escaped_file_path = file_path.replace('.dot', '_escaped.dot')
    with open(escaped_file_path, 'w', encoding='utf-8') as file:
        file.write(escaped_content)

    print(f"Escaped file saved as: {escaped_file_path}")

# Example usage:
# Assuming your .dot file is named 'cpg_graph.dot'
file_path = 'export.dot'
escape_nested_quotes_in_file(file_path)
