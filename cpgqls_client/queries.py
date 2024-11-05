
def import_code_query(path, project_name=None, language=None):
    if not path:
        raise Exception('An importCode query requires a project path')
    if project_name and language:
        fmt_str = u"""importCode(inputPath=\"%s\", projectName=\"%s\",language=\"%s\")"""
        return fmt_str % (path, project_name, language)
    if project_name and (language is None):
        fmt_str = u"""importCode(inputPath=\"%s\")"""
        return fmt_str % path
    return u"importCode(\"%s\")" % path


def open_query(project_name):
    return f"open(\"{project_name}\")"


def close_query(project_name):
    return f"close(\"{project_name}\")"


def delete_query(project_name):
    return f"delete(\"{project_name}\")"


def help_query():
    return f"help"


def workspace_query():
    return "workspace"


def project_query():
    return "project"

def export_cpg_as_json(client, cpg_type="cpg"):
    """
    获取 Joern 生成的 CPG 的 JSON 格式
    :param client: 连接到 Joern 的客户端对象
    :param cpg_type: 要获取的图类型，默认为 "cpg"。可选值为 "ast", "cfg", "dfg" 等
    :return: CPG 的 JSON 格式字符串
    """
    if cpg_type not in ["Cpg", "Ast", "Cfg", "Ddg", "Pdg", "Cpg14"]:
        raise ValueError("Invalid CPG type. Choose from 'Cpg', 'Ast', 'Cfg', 'Ddg', 'Pdg', 'Cpg14'.")

    # 导出到本地的地址
    query = f"cpg.all.toJsonPretty #> \"/home/kevin/joern-parse/out.json\" "

    try:
        result = client.execute(query)
        return result
    except Exception as e:
        print(f"Error while fetching the {cpg_type} in JSON format: {e}")
        return None

def export_cpg_as_dot(client, dot_type="Cfg", file_name = "cpg.dot"):
    """
    获取 Joern 生成的 CPG 的 Dot 格式
    :param file_name:
    :param dot_type:
    :param client: 连接到 Joern 的客户端对象
    :param dot_type: 要获取的图类型，默认为 "cpg"。可选值为 "ast", "cfg", "dfg" 等
    :return: CPG 的 Dot 格式字符串
    """
    if dot_type not in ["Cpg", "Ast", "Cfg", "Ddg", "Pdg", "Cpg14"]:
        raise ValueError("Invalid CPG type. Choose from 'Cpg', 'Ast', 'Cfg', 'Ddg', 'Pdg', 'Cpg14'.")

    # 使用 Joern 查询语句获取指定类型的 CPG
    # query = f"cpg.method(\"main\").dot{dot_type}.l #> \"/home/fzc/workspace/DeepRace/tests/cpg_process/{dot_type}.dot\" "
    query = f"cpg.method(\"main\").dot{dot_type}.l #> \"/home/kevin/joern-parse/{file_name}\" "

    try:
        result = client.execute(query)
        return result
    except Exception as e:
        print(f"Error while fetching the {dot_type} in JSON format: {e}")
        return None


def export_callgraph_as_dot(client):
    """
    获取 Joern 生成的 CallGraph 格式
    """
    # 使用 Joern 查询语句获取指定类型的 CPG
    # query = f"cpg.method(\"main\").dot{dot_type}.l #> \"/home/fzc/workspace/DeepRace/tests/cpg_process/{dot_type}.dot\" "
    query = f"cpg.dotCallGraph #> \"/home/kevin/joern-parse/callgraph.dot\" "

    try:
        result = client.execute(query)
        return result
    except Exception as e:
        print(f"Error while fetching the callgraph: {e}")
        return None