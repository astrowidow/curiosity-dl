import os
import subprocess


def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret_txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.input_variables:
        ret_txt += dot_edge.format(id(x), id(f))
    ret_txt += dot_edge.format(id(f), id(f.output_variable()))
    return ret_txt


def get_dot_graph(output, verbose=False):
    ret_txt = ''
    funcs = []
    pushed_func_set = set()

    def add_func(f):
        if f not in pushed_func_set:
            funcs.append(f)
            pushed_func_set.add(f)

    ret_txt += _dot_var(output)
    add_func(output.creator)
    while funcs:
        func = funcs.pop()
        ret_txt += _dot_func(func)
        for x in func.input_variables:
            ret_txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)

    ret_txt = 'digraph g {\n' + ret_txt + '}'
    return ret_txt


def plot_dot_graph(output, verbose=False, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # saves dot data
    save_dir = os.path.join(os.path.dirname(__file__), '../graph_img')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    graph_script_path = os.path.join(save_dir, 'tmp_graph.dot')
    with open(graph_script_path, 'w') as f:
        f.write(dot_graph)

    # calls dot command
    extension = os.path.splitext(to_file)[1][1:]
    graph_img_path = os.path.join(save_dir, to_file)
    cmd = 'dot {} -T {} -o {}'.format(graph_script_path, extension, graph_img_path)
    subprocess.run(cmd, shell=True)
