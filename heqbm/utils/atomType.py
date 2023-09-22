from heqbm.utils import DataDict

def get_type_from_name(name: str) -> int:
    atom_name = ''.join(i for i in name.split(DataDict.STR_SEPARATOR)[-1] if not i.isdigit())
    if atom_name.startswith('H'):
        return 1
    if atom_name.startswith('C'):
        return 6
    if atom_name.startswith('N'):
        return 7
    if atom_name.startswith('O'):
        return 8
    if atom_name.startswith('S'):
        return 16
    return 0