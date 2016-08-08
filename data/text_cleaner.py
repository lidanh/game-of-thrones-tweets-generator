import re
import sys

def remove_url_from_file(src_file, dst_file):
	"""
	Remove URLs from the file, keep everything else as it was
	"""
	with open(src_file, 'r') as f:
		with open (dst_file, 'wa') as outfile:
			for line in f:
				out_string = remove_url(line)
				outfile.write(out_string)


def remove_url(str):
	return re.sub(r'https?:\/\/.*', '', str, flags=re.MULTILINE)

def clean_non_ascii(str):
	"""
	remove non ascii chars from a string
	"""
	str = ''.join([i if ord(i) < 128 else ' ' for i in str])
	return str


def remove_newline(str):
	"""
	replace newline with single space
	"""
	return str.replace('\n', ' ')


def clean_text(str):
	str = clean_non_ascii(str)
	str = remove_url(str)
	str = remove_newline(str)
	return str


def main():
	infile = sys.argv[1]
	outfile = sys.argv[2]
	remove_url_from_file(infile, outfile)

if __name__ == '__main__':
	main()
