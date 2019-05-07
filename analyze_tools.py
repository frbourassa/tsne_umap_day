# -*- coding:utf-8 -*-
"""
Module containing some basic functions to simplify the analyze_data notebook.
@author: frbourassa
May 7, 2019
"""

import os
from format_tools import load_object

def list_available(folder, condition=None):
    """ A function to list the available files in a folder that return True
    when the one-argument function condition is applied on them.

    Args:
        folder (str): the folder to list, either relative to the
            current working directory, or absolute.
        condition (func): a one-argument function returning a bool
            that will be applied to filter the file names in the folder.
            Only file names for which condition(name) gives True are included.

    Returns:
        available_files (dict): dictionary of available files
    """
    current_working_dir = os.getcwd()
    print("The current working directory is ", current_working_dir)

    # If no condition was given, include all files
    if condition is None:
        condition = lambda x: True

    # Build a dictionary of available files.
    list_available_files = [fi for fi in sorted(os.listdir(folder))
                                if condition(fi)]
    available_files = {i:list_available_files[i]
                            for i in range(len(list_available_files))}

    # Print the dictionary
    print("There are {0} available .pkl files in {1}: {{".format(len(list_available_files), folder))
    for i in range(len(available_files)):
        print('\t{0}:"{1}"'.format(i, available_files[i]))
    print("}")
    print("Now, select your file in the cell below")

    return available_files

def load_chosen(chosen_idx, folder, availables):
    """ A function to load the file corresponding to the key chosen_key
    in the dictionary availables of available files in folder.

    Args:
        chosen_idx (int): the key corresponding to the desired file
            in availables.
        folder (str): the folder in which the chosen file is
        availables (dict): the kind of dict returned by list_available

    Returns:
        (obj): the object stored in the chosen file, loaded with pickle,
            if it's possible to load it; otherwise, returns None.
    """
    try:
        file_name = availables[chosen_idx]
    except KeyError:
        print("The index {} is out of range; look again at the list above and select a valid key.".format(file_chosen_index))
        file_name = ""
        return None
    else:
        file_path = os.path.join(folder, file_name)
        print("Will try to import:\n{}".format(file_path))

    try:
        df = load_object(file_path)
    except FileNotFoundError as e:  # inexisting file
        raise e
    except Exception:  # not the right kind of file
        df = None
        raise TypeError("Could not load that file; try again.")
    else:
        print("\nSuccesfully loaded the following object: \n")
        print(df)
        return df
