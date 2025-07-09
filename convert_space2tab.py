import os

def convert_spaces_to_tabs(file_path, tab_size=4):
	with open(file_path, 'r') as f:
		lines = f.readlines()

	converted = []
	for line in lines:
		# Only replace leading spaces, preserve inline spacing
		leading_spaces = len(line) - len(line.lstrip(' '))
		if leading_spaces > 0 and leading_spaces % tab_size == 0:
			tabs = '\t' * (leading_spaces // tab_size)
			line = tabs + line.lstrip(' ')
		converted.append(line)

	with open(file_path, 'w') as f:
		f.writelines(converted)

def convert_all_py_files(root_dir):
	for dirpath, _, filenames in os.walk(root_dir):
		for filename in filenames:
			if filename.endswith('.py'):
				full_path = os.path.join(dirpath, filename)
				try:
					convert_spaces_to_tabs(full_path)
					print(f"Converted: {full_path}")
				except Exception as e:
					print(f"Failed to convert {full_path}: {e}")

if __name__ == "__main__":
	import sys
	if len(sys.argv) != 2:
		print("Usage: python convert_to_tabs.py /path/to/your/code")
	else:
		convert_all_py_files(sys.argv[1])
