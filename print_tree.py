import os

def print_tree(startpath, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = {'.git', '.idea', '__pycache__', '.ipynb_checkpoints', 'venv', 'env'}

    for root, dirs, files in os.walk(startpath):
        # 过滤掉不需要显示的文件夹
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            # 过滤掉一些特定的系统文件或临时文件
            if f.startswith('.') or f.endswith('.pyc'):
                continue
            print(f'{subindent}{f}')

if __name__ == '__main__':
    # 获取当前脚本所在的目录
    current_dir = os.getcwd()
    print(f"Project Structure for: {current_dir}\n")
    print_tree(current_dir)