from global_funcs import load_data_config
from dask.distributed import Client
from dask_cuda import LocalCUDACluster


if __name__ == '__main__':
    import dask_cudf

    print("loading configs...")
    configs = load_data_config()

    print("setting variables...")
    clean_fasta_filepath = configs['clean_fasta_file']
    output_dir = configs['output_dir']
    project_name = configs['project_name']
    unq_labs_dir = configs['unq_labs_dir']
    unq_labs_dir_csv = configs['unq_labs_dir_csv']
    data_dir = configs['data_dir']
    label_col_name = configs['label_col_name']
    label_regex = configs['label_regex']
    dask_dir = configs['dask_dir']
    
    print("starting dask cluster...")
    cluster = LocalCUDACluster(local_directory=dask_dir)
    client = Client(cluster)
    print("finished starting dask cluster.")

    
    def extract_labels(df):
        df[label_col_name] = df[label_col_name].str.extract(label_regex).loc[:, 0]
        return df
    
    print("reading data...")
    df = dask_cudf.read_parquet(clean_fasta_filepath).repartition(partition_size="10M")
    print("extracting all labels...")
    df = df.map_partitions(extract_labels)
    
    
    print("getting unique labels...")
    unq_labs_df = df.sort_values(label_col_name)[label_col_name].unique().to_frame()
    print(f"saving unique labels to:\n{unq_labs_dir}")
    _ = unq_labs_df.to_parquet(unq_labs_dir)
    print(f"saving unique labels to:\n{unq_labs_dir_csv}")
    _ = unq_labs_df.to_csv(unq_labs_dir_csv, index=False, single_file=True)
    
    print("encoding labels...")
    df = df.categorize(columns=[label_col_name])
    df[label_col_name] = df[label_col_name].cat.codes
    
    print(f"saving updated data to:\n{data_dir}")
    _ = df.to_parquet(data_dir)
    
    print("deleting dask df...")
    del df, unq_labs_df

    print("shutting down...")
    client.shutdown()
    client.close()
    print("label extraction complete!")