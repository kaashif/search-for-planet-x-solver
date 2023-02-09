import sys
import json

filenames = sys.argv[1:]

# Note that the letter of the project doesn't matter, the research project clues are
# specific to the name of the project. e.g. A: Gas Clouds and F: Gas Clouds are both
# the same project with the same clues.
project_name_to_result_list = {}

def extract_research(lines: list[str]):
    letter_name_clue = []
    line_index = 0

    while True:
        if lines[line_index].startswith("X1"):
            break
        line_index += 1

    def non_whitespace_words(line):
        return [
            word.strip()
            for word in line.split(" ")
            if len(word.strip()) > 0
        ]


    def extract_project_name() -> str:
        nonlocal line_index
        start_line = lines[line_index]
        words = non_whitespace_words(start_line)

        # If the line ends in any of these, there are more words on the next two lines.
        if words[-1] in ["&", "Dwarf", "Gas"]:
            line_index += 2
            words += non_whitespace_words(lines[line_index])

        return " ".join(words[1:])

    def extract_clue() -> str:
        nonlocal line_index
        clue_lines = []

        while True:
            line = lines[line_index]
            clue_lines.append(line.rstrip())
            if line.endswith(".\n"):
                break
            line_index += 1

        return " ".join(clue_lines)

    def skip_all_whitespace_or_single_letter():
        nonlocal line_index
        if len(lines[line_index].strip()) <= 1:
            line_index += 1

    # The reason we do this rather than a loop is that the change in the line index is different,
    # and that the order is:
    # X1 name/clue, A name/clue, B name/clue, C name/clue, F/E/D names, F/E/D clues
    x1_name = extract_project_name()
    line_index += 2
    x1_clue = extract_clue()

    line_index += 4
    a_name = extract_project_name()
    line_index += 2
    a_clue = extract_clue()

    line_index += 4
    b_name = extract_project_name()
    line_index += 1
    skip_all_whitespace_or_single_letter()
    b_clue = extract_clue()

    line_index += 4
    c_name = extract_project_name()
    line_index += 2
    c_clue = extract_clue()

    line_index += 4
    f_name = extract_project_name()

    line_index += 2
    e_name = extract_project_name()

    line_index += 2
    d_name = extract_project_name()

    line_index += 2
    f_clue = extract_clue()

    line_index += 3
    e_clue = extract_clue()

    line_index += 3
    d_clue = extract_clue()

    letter_name_clue.append(["X1", x1_name, x1_clue])
    letter_name_clue.append(["A", a_name, a_clue])
    letter_name_clue.append(["B", b_name, b_clue])
    letter_name_clue.append(["C", c_name, c_clue])
    letter_name_clue.append(["D", d_name, d_clue])
    letter_name_clue.append(["E", e_name, e_clue])
    letter_name_clue.append(["F", f_name, f_clue])

    return letter_name_clue

research = []

for filename in filenames:
    print(filename)
    new_research = extract_research(open(filename, "r").readlines())
    print(new_research)
    research.append(new_research)

print(json.dumps(research))
