import pandas as pd

def get_data(filename, encoding='iso-8859-1'):
    raw = pd.read_csv(filename, encoding=encoding)
    return raw

def split(all, omit):
    valida, omits, training_mask = all, pd.DataFrame(), all.columns

    if omit:
        omits = all[omit]
        training_mask = list(filter(lambda h: h not in omit, all.columns))
        valida = all[training_mask]

    return valida, omits, training_mask

