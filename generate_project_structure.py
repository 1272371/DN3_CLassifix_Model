import os


def generate_folder_structure():
    root_dir = os.getcwd()
    excluded_dirs = (".git", ".vscode", "__pycache__")
    excluded_files = (".gitignore", "settings.json")

    folder_structure = ""

    for root, dirs, files in os.walk(root_dir):
        # Exclude directories and files.
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        files[:] = [f for f in files if f not in excluded_files]

        current_dir = os.path.relpath(root, root_dir)

        if current_dir == ".":
            current_dir = "project"

        folder_structure += f"{current_dir}/\n"
        indent = "    " * (current_dir.count("/"))

        for i, file in enumerate(files):
            if i < len(files) - 1:
                folder_structure += f"{indent}├── {file}\n"
            else:
                folder_structure += f"{indent}└── {file}\n"

    with open("folder_structure.txt", "w", encoding="utf-8") as file:
        file.write(folder_structure)

    print("Folder structure generated successfully.")


# Call the function to generate the folder structure and create the text file
generate_folder_structure()
