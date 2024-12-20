import signal
import time
import random
import libcst as cst
import libcst.matchers as m
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig, Part
from vertexai.preview.tuning import sft

def handler(signum, frame):
    raise Exception("end of time")

def create_gemini_config(message, temperature = 0.8):
    assert isinstance(message,str)
    config = {
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": message},
        ],
    }
    return config

def request_gemini_engine(config,gemini_version):
    ret = None
    if gemini_version=='original':
        gemini_model = GenerativeModel("gemini-1.5-pro-002", generation_config=GenerationConfig(temperature=config['temperature']))
    else:
        tuning_job = sft.SupervisedTuningJob(gemini_version)
        gemini_model = GenerativeModel(tuning_job.tuned_model_endpoint_name, generation_config=GenerationConfig(temperature=config['temperature']))
    success = False
    exec_count = 0
    while not success and exec_count<10:
        exec_count += 1
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = gemini_model.generate_content(
                        contents=[
                            config['messages'][0]["content"]
                        ],
                        generation_config=GenerationConfig(
                            temperature=config['temperature'],
                            top_p=0.0,
                            top_k=1,
                            candidate_count=1,
                            max_output_tokens=4096,
                        ),
                    )
            assert isinstance(ret.text,str)
            ret = {
                'id': 'gemini',
                'content': [{'text': ret.text,'type': 'text'}],
                'model': "tuned_gemini",
                'role': 'assistant', 'stop_reason': 'end_turn', 'stop_sequence': None, 'type': 'message',
                'usage': {'input_tokens': ret.usage_metadata.prompt_token_count,
                          'output_tokens': ret.usage_metadata.candidates_token_count}
            }
            success = True
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            time.sleep(3)
    return ret

def filter_none_python(structure):
    for key, value in list(structure.items()):
        if (
            not "functions" in value.keys()
            and not "classes" in value.keys()
            and not "text" in value.keys()
        ) or not len(value.keys()) == 3:
            filter_none_python(value)

            if structure[key] == {}:
                del structure[key]
        else:
            if not key.endswith(".py"):
                del structure[key]

def filter_out_test_files(structure):
    for key, value in list(structure.items()):
        if key.startswith("test"):
            del structure[key]
        elif isinstance(value, dict):
            filter_out_test_files(value)

def show_project_structure(structure, spacing=0) -> str:
    pp_string = ""
    items = list(structure.items())
    random.shuffle(items)
    for key, value in items:
        if "." in key and ".py" not in key:
            continue  # skip none python files
        if key=='test' or key=='tests':
            continue
        if "." in key:
            pp_string += " " * spacing + str(key) + "\n"
        else:
            pp_string += " " * spacing + str(key) + "/" + "\n"
        if "classes" not in value:
            pp_string += show_project_structure(value, spacing + 4)
    return pp_string


class CompressTransformer(cst.CSTTransformer):
    DESCRIPTION = str = "Replaces function body with ..."
    replacement_string = '"$$FUNC_BODY_REPLACEMENT_STRING$$"'

    def __init__(self, keep_constant=True):
        self.keep_constant = keep_constant

    def leave_Module(
            self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        new_body = [
            stmt
            for stmt in updated_node.body
            if m.matches(stmt, m.ClassDef())
               or m.matches(stmt, m.FunctionDef())
               or (
                       self.keep_constant
                       and m.matches(stmt, m.SimpleStatementLine())
                       and m.matches(stmt.body[0], m.Assign())
               )
        ]
        return updated_node.with_changes(body=new_body)

    def leave_ClassDef(
            self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        # Remove docstring in the class body
        new_body = [
            stmt
            for stmt in updated_node.body.body
            if not (
                    m.matches(stmt, m.SimpleStatementLine())
                    and m.matches(stmt.body[0], m.Expr())
                    and m.matches(stmt.body[0].value, m.SimpleString())
            )
        ]
        return updated_node.with_changes(body=cst.IndentedBlock(body=new_body))

    def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        new_expr = cst.Expr(value=cst.SimpleString(value=self.replacement_string))
        new_body = cst.IndentedBlock((new_expr,))
        # another way: replace with pass?
        return updated_node.with_changes(body=new_body)


def get_full_file_paths_and_classes_and_functions(structure, current_path=""):
    files = []
    classes = []
    functions = []
    for name, content in structure.items():
        if isinstance(content, dict):
            if (
                    not "functions" in content.keys()
                    and not "classes" in content.keys()
                    and not "text" in content.keys()
            ) or not len(content.keys()) == 3:
                next_path = f"{current_path}/{name}" if current_path else name
                (
                    sub_files,
                    sub_classes,
                    sub_functions,
                ) = get_full_file_paths_and_classes_and_functions(content, next_path)
                files.extend(sub_files)
                classes.extend(sub_classes)
                functions.extend(sub_functions)
            else:
                next_path = f"{current_path}/{name}" if current_path else name
                files.append((next_path, content["text"]))
                if "classes" in content:
                    for clazz in content["classes"]:
                        classes.append(
                            {
                                "file": next_path,
                                "name": clazz["name"],
                                "start_line": clazz["start_line"],
                                "end_line": clazz["end_line"],
                                "methods": [
                                    {
                                        "name": method["name"],
                                        "start_line": method["start_line"],
                                        "end_line": method["end_line"],
                                    }
                                    for method in clazz.get("methods", [])
                                ],
                            }
                        )
                if "functions" in content:
                    for function in content["functions"]:
                        function["file"] = next_path
                        functions.append(function)
        else:
            next_path = f"{current_path}/{name}" if current_path else name
            files.append(next_path)
    return files, classes, functions


def get_repo_files(structure, filepaths: list[str], **kwargs):
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    file_contents = dict()
    for filepath in filepaths:
        content = None
        for file_content in files:
            if file_content[0] == filepath:
                content = "\n".join(file_content[1])
                file_contents[filepath] = content
                break
    return file_contents

file_content_in_block_template = """
### File: {file_name} ###
{file_content}
"""

def get_skeleton(raw_code, keep_constant: bool = True):
    try:
        tree = cst.parse_module(raw_code)
    except:
        return raw_code

    transformer = CompressTransformer(keep_constant=keep_constant)
    modified_tree = tree.visit(transformer)
    code = modified_tree.code
    code = code.replace(CompressTransformer.replacement_string + "\n", "...\n")
    code = code.replace(CompressTransformer.replacement_string, "...\n")
    return code

def get_compressed_content(file_names, structure):
    file_contents = get_repo_files(structure, file_names)
    compressed_file_contents = {fn: get_skeleton(code) for fn, code in file_contents.items()}
    contents = [
        file_content_in_block_template.format(file_name=fn, file_content=code)
        for fn, code in compressed_file_contents.items()
    ]
    return "".join(contents)