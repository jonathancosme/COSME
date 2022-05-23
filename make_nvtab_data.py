from global_funcs import load_program_config
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import rmm
from nvtabular.utils import device_mem_size
import shutil
import pathlib

if __name__ == '__main__':
    
    import nvtabular as nvt
    
    print("loading configs...")
    configs = load_program_config()
    
    print("setting variables...")
    output_dir = configs['output_dir']
    project_name = configs['project_name']
    input_col_name = configs['input_col_name']
    label_col_name = configs['label_col_name']
    random_seed = configs['random_seed']
    data_splits = configs['data_splits']
    base_col_names = configs['base_col_names']
    
    base_in_pathname = f"{output_dir}/{project_name}/data/{project_name}"
    base_out_pathname = f"{output_dir}/{project_name}/nvtab"
    
    # define some information about where to get our data
    dask_workdir = pathlib.Path(base_out_pathname, "dask", "workdir")
    stats_path = pathlib.Path(base_out_pathname, "dask", "stats")

    # Make sure we have a clean worker space for Dask
    if pathlib.Path.is_dir(dask_workdir):
        shutil.rmtree(dask_workdir)
    dask_workdir.mkdir(parents=True)

    # Make sure we have a clean stats space for Dask
    if pathlib.Path.is_dir(stats_path):
        shutil.rmtree(stats_path)
    stats_path.mkdir(parents=True)

    # Get device memory capacity
    capacity = device_mem_size(kind="total")
    
    # Deploy a Single-Machine Multi-GPU Cluster
    protocol = "tcp"  # "tcp" or "ucx"
    visible_devices = "0"  # Delect devices to place workers
    device_spill_frac = 0.5  # Spill GPU-Worker memory to host at this limit.
    # Reduce if spilling fails to prevent
    # device memory errors.
    cluster = None  # (Optional) Specify existing scheduler port
    if cluster is None:
        cluster = LocalCUDACluster(
            protocol=protocol,
            CUDA_VISIBLE_DEVICES=visible_devices,
            local_directory=dask_workdir,
            device_memory_limit=capacity * device_spill_frac,
        )

    # Create the distributed client
    client = Client(cluster)
    client
    
    # Initialize RMM pool on ALL workers
    def _rmm_pool():
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=None,  # Use default size
        )


    client.run(_rmm_pool)
    
    print("creating nvtab workflow...")
    cat_features = [input_col_name] >> nvt.ops.Categorify() >>  nvt.ops.ListSlice(start=0, end=150, pad=True, pad_value=0)
    output = cat_features + label_col_name
    workflow = nvt.Workflow(output)

    print("fitting nvtab workflow on training data...")
    workflow.fit(nvt.Dataset(f"{base_in_pathname}_train", engine='parquet', row_group_size=10000))
    
    print("saving fitting nvtab workflow...")
    workflow.save(f"{base_out_pathname}/workflow")
    
    shuffle= nvt.io.Shuffle.PER_PARTITION
    
    print("making nvtab dataset for training...")
    workflow.transform(nvt.Dataset(f"{base_in_pathname}_train", engine='parquet', row_group_size=10000)).to_parquet(
        output_path=f"{base_out_pathname}/train",
        shuffle=shuffle,
        cats=[input_col_name],
        labels=[label_col_name],
    )
    
    print("making nvtab dataset for validation...")
    workflow.transform(nvt.Dataset(f"{base_in_pathname}_val", engine='parquet', row_group_size=10000)).to_parquet(
        output_path=f"{base_out_pathname}/val",
        shuffle=None,
        out_files_per_proc=None,
        cats=[input_col_name],
        labels=[label_col_name],
    )
    
    print("making nvtab dataset for testing...")
    workflow.transform(nvt.Dataset(f"{base_in_pathname}_test", engine='parquet', row_group_size=10000)).to_parquet(
        output_path=f"{base_out_pathname}/test",
        shuffle=None,
        out_files_per_proc=None,
        cats=[input_col_name],
        labels=[label_col_name],
    )
    
    print("shutting down...")
    client.shutdown()
    cluster.close()
    print("nvtabular datasets creation complete!")