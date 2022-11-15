from tdc.single_pred import ADME

data = ADME(name = 'Caco2_Wang')
split = data.get_split()
print(split)