from global_funcs import load_raw_data_config
from dask.distributed import Client
from dask_cuda import LocalCUDACluster


if __name__ == '__main__':
    print("starting dask cluster...")
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print("finished starting dask cluster.")

    import dask_cudf

    print("loading configs...")
    configs = load_raw_data_config()

    print("setting variables...")
    raw_fasta_file = configs['raw_fasta_file']
    clean_fasta_file = configs['clean_fasta_file']
    base_col_names =  configs['base_col_names']
    fasta_sep = configs['fasta_sep']

    print("cleaning and saving fasta data...")
    df = dask_cudf.read_csv(raw_fasta_file, sep=fasta_sep, names=base_col_names, dtype=str)
    df['label'] = df['label'].shift()
    df = df.dropna().reset_index(drop=True)

    _ = df.to_parquet(clean_fasta_file)

    print("deleting dask df...")
    del df

    print("shutting down...")
    client.shutdown()
    client.close()
    print("fasta cleaning complete!")