import torch

vocabulaire = []
X_train, Y_train = torch.tensor(0), torch.tensor(0)
X_val, Y_val = torch.tensor(0), torch.tensor(0)

def build_data(data_dir, context_len, train_split=0.8):
    global vocabulaire
    global X_train, Y_train, X_val, Y_val

    # load les données
    fichier = open(data_dir)
    donnees = fichier.read()
    villes = donnees.replace('\n', ',').split(',')

    # preparation des données
    # on rajoute le token . au début et en fin
    for ville, i in zip(villes, range(len(villes))):
        villes[i] = ville + '.'

    # création du vocabulaire
    for ville in villes:
        for c in ville:
            if c not in vocabulaire:
                vocabulaire.append(c)

    vocabulaire = sorted(vocabulaire)
    vocabulaire[0] = '.'
    vocabulaire[3] = " "

    # pour convertir char <-> int
    char_to_int = {}
    int_to_char = {}

    for (c, i) in zip(vocabulaire, range(len(vocabulaire))):
        char_to_int[c] = i
        int_to_char[i] = c

    # création du dataset
    X = []
    Y = []

    for ville in villes:
        context = [0] * context_len

        for ch in ville:
            X.append(context)
            Y.append(char_to_int[ch])

            context = context[1:] + [char_to_int[ch]]

    X = torch.tensor(X) # (M, context_len), int64
    Y = torch.tensor(Y) # (M), int64

    n1 = int(train_split*X.shape[0])

    X_train = X[:n1]
    X_val = X[n1:]

    Y_train = Y[:n1]
    Y_val = Y[n1:]

def get_batch(batch_size, split, device):
    global X_train, Y_train, X_val, Y_val

    if split == 'train':
        ix = torch.randint(X_train.shape[0], (batch_size,))

        if device == 'cuda':
            Xb = X_train[ix].pin_memory().to(device, non_blocking=True)
            Yb = Y_train[ix].pin_memory().to(device, non_blocking=True)
        else:
            Xb = X_train[ix].to(device)
            Yb = Y_train[ix].to(device)
    else:
        ix = torch.randint(X_val.shape[0], (batch_size,))

        if device == 'cuda':
            Xb = X_val[ix].pin_memory().to(device, non_blocking=True)
            Yb = Y_val[ix].pin_memory().to(device, non_blocking=True)
        else:
            Xb = X_val[ix].to(device)
            Yb = Y_val[ix].to(device)
    
    return Xb, Yb

def get_voc_size():
    global vocabulaire
    return len(vocabulaire)