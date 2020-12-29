import fileinput
import os

def fix_comma_issue(filename):
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace(',', '.').replace('\n', ''))

if __name__ == '__main__':

    files = os.listdir('labels/')
    for file in files:
        fix_comma_issue('labels/' + file)