def replace_words_in_file(file_path, replacements, save_path=None):
    """
    Reads a file, replaces specific words, and saves it.
    
    :param file_path: Path to the original file.
    :param replacements: Dictionary where key is the word to be replaced and value is the new word.
    :param save_path: Path to save the modified file. If None, it overwrites the original file.
    """
    # Open and read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace words based on the replacements dictionary
    for old_word, new_word in replacements.items():
        content = content.replace(old_word, new_word)

    # Determine the save path
    if save_path is None:
        save_path = file_path  # Overwrite the original file if no new path is provided

    # Write the modified content to the file
    with open(save_path, 'w') as file:
        file.write(content)