import sys
import json
import re

filenames = sys.argv[1:]

# Note that the letter of the project doesn't matter, the research project clues are
# specific to the name of the project. e.g. A: Gas Clouds and F: Gas Clouds are both
# the same project with the same clues.
project_name_to_result_list = {}


def non_whitespace_words(line):
    return [
        word.strip()
        for word in line.split(" ")
        if len(word.strip()) > 0
    ]


def line_has_name(line):
    name_regex = r"^[A-Z]1?:"
    return re.match(name_regex, line)


def line_has_clue(line):
    words = non_whitespace_words(line)
    return not line_has_name(line) and len(words) >= 2


def line_is_interesting(line):
    return line_has_name(line) or line_has_clue(line)


def extract_research(lines: list[str]):
    letter_name_clue = []
    line_index = 0

    def extract_project_name():
        nonlocal line_index
        start_line = lines[line_index]
        words = non_whitespace_words(start_line)
        letter = words[0][:-1]

        # If the line ends in any of these, there are more words on the next two lines.
        if words[-1] in ["&", "Dwarf", "Gas"]:
            line_index += 2
            words += non_whitespace_words(lines[line_index])

        line_index += 1
        return letter, " ".join(words[1:])

    def extract_clue() -> str:
        nonlocal line_index
        clue_lines = []

        while True:
            line = lines[line_index]
            clue_lines.append(line.rstrip())
            line_index += 1
            if line.endswith(".\n"):
                break

        return " ".join(clue_lines)

    def skip_until(cond):
        nonlocal line_index
        while not cond(lines[line_index]):
            line_index += 1

    # The pdfminer output looks like:
    # X1 name/clue, A name/clue, B name/clue, C name/clue, F/E/D names, F/E/D clues

    completed_letters = set()
    letter_and_name = []

    skip_until(line_has_name)
    while len(completed_letters) < 7:
        skip_until(line_is_interesting)

        line = lines[line_index]
        if line_has_name(line):
            letter, name = extract_project_name()
            letter_and_name.append((letter, name))

        elif line_has_clue(line):
            letter, name = letter_and_name.pop(0)
            clue = extract_clue()
            letter_name_clue.append([letter, name, clue])
            completed_letters.add(letter)

    return letter_name_clue


research = []

for filename in filenames:
    new_research = extract_research(open(filename, "r").readlines())
    research.append(new_research)

print(json.dumps(research))
