from sklearn.preprocessing import MinMaxScaler, StandardScaler

def scale_datasets(datasets, method='minmax', return_scaler=False):
    """
    Escala m�ltiples datasets usando el tipo de escalador indicado,
    ajustando sobre el primer dataset.

    Par�metros:
    -----------
    datasets : list or tuple of np.ndarray
        Conjuntos de datos a escalar. El primero se usar� para ajustar el scaler.
    
    method : str
        Tipo de escalado: 'minmax' o 'standard'.
    
    return_scaler : bool
        Si True, tambi�n devuelve el scaler ajustado.

    Retorna:
    --------
    Lista de datasets escalados (y opcionalmente el scaler)
    """
    if not datasets:
        raise ValueError("No se proporcionaron datasets para escalar.")

    # Selecci�n del escalador
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("M�todo de escalado no v�lido. Usa 'minmax' o 'standard'.")

    # Ajustar sobre el primer dataset y transformar todos
    scaled = [scaler.fit_transform(datasets[0])]
    for X in datasets[1:]:
        scaled.append(scaler.transform(X))

    if return_scaler:
        return scaled, scaler
    return scaled