import nibabel as nib
import numpy as np
import os
import PIL.Image as Image
from scipy.ndimage import affine_transform, map_coordinates
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import signal
import sys

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    print('\nReceived interrupt signal, exiting...')
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def resample_to_target(source_data, source_affine, target_affine, target_shape):
    """Resample source data to target space"""
    # Calculate transformation matrix from target to source coordinate system
    transform_matrix = np.dot(np.linalg.inv(source_affine), target_affine)
    
    # Create coordinate grid for target space
    i, j, k = np.mgrid[0:target_shape[0], 0:target_shape[1], 0:target_shape[2]]
    coords = np.stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
    
    # Transform target coordinates to source coordinate system
    source_coords = np.dot(transform_matrix, coords)[:3]
    source_coords = source_coords.reshape(3, *target_shape)
    
    # Use map_coordinates for interpolation
    resampled_data = map_coordinates(
        source_data, 
        source_coords, 
        order=1,  # Linear interpolation
        mode='constant', 
        cval=0
    )
    
    return resampled_data

def create_common_space(affine1, shape1, affine2, shape2):
    """Create a common world coordinate system"""
    # Get boundaries of both images in world coordinate system
    def get_world_bounds(affine, shape):
        corners = np.array([
            [0, 0, 0, 1],
            [shape[0]-1, 0, 0, 1],
            [0, shape[1]-1, 0, 1],
            [0, 0, shape[2]-1, 1],
            [shape[0]-1, shape[1]-1, 0, 1],
            [shape[0]-1, 0, shape[2]-1, 1],
            [0, shape[1]-1, shape[2]-1, 1],
            [shape[0]-1, shape[1]-1, shape[2]-1, 1]
        ])
        world_corners = np.dot(corners, affine.T)[:, :3]
        return world_corners.min(axis=0), world_corners.max(axis=0)
    
    min1, max1 = get_world_bounds(affine1, shape1)
    min2, max2 = get_world_bounds(affine2, shape2)
    
    # Calculate common boundaries
    common_min = np.minimum(min1, min2)
    common_max = np.maximum(max1, max2)
    
    # Choose smaller voxel size as target resolution
    voxel_size1 = np.sqrt(np.sum(affine1[:3, :3]**2, axis=0))
    voxel_size2 = np.sqrt(np.sum(affine2[:3, :3]**2, axis=0))
    target_voxel_size = np.minimum(voxel_size1, voxel_size2)
    
    # Create target space shape
    target_shape = np.ceil((common_max - common_min) / target_voxel_size).astype(int)
    
    # Create target affine matrix (axis-aligned)
    target_affine = np.eye(4)
    target_affine[:3, :3] = np.diag(target_voxel_size)
    target_affine[:3, 3] = common_min
    
    return target_affine, target_shape

def process_single_pair(args):
    """Process a single T1-T2 pair for multiprocessing"""
    file1, file2, dirname, output_dir = args
    
    try:
        # Check filename consistency
        filename1 = file1.split("_")[:-1]
        filename2 = file2.split("_")[:-1]
        if filename1 != filename2:
            return f"Error: {file1} and {file2} filename mismatch"
        
        # Load images
        img1 = nib.load(os.path.join(dirname[0], file1))
        img2 = nib.load(os.path.join(dirname[1], file2))
        
        # Get data and affine matrices
        data1 = img1.get_fdata()
        data2 = img2.get_fdata()
        affine1 = img1.affine
        affine2 = img2.affine
        
        # Create common coordinate system
        target_affine, target_shape = create_common_space(affine1, data1.shape, affine2, data2.shape)
        
        # Resample
        data1_resampled = resample_to_target(data1, affine1, target_affine, target_shape)
        data2_resampled = resample_to_target(data2, affine2, target_affine, target_shape)
        
        # Save results
        nib.save(nib.Nifti1Image(data1_resampled, target_affine), os.path.join(output_dir[0], file1))
        nib.save(nib.Nifti1Image(data2_resampled, target_affine), os.path.join(output_dir[1], file2))
        
        return f"Success: {file1}"
        
    except KeyboardInterrupt:
        print("\nUser interrupted, cleaning up...")
        sys.exit(0)
    except Exception as e:
        return f"Error {file1}: {str(e)}"

if __name__=="__main__":
    dirname=["./IXI-T1/","./IXI-T2/"]
    filelist_t1 = sorted(os.listdir(dirname[0]))
    filelist_t2 = sorted(os.listdir(dirname[1]))
    
    len_t1 = len(filelist_t1)
    len_t2 = len(filelist_t2)
    if len_t1 != len_t2:
        raise ValueError("T1 and T2 file counts do not match")
    
    output_dir = ["./IXI-T1-Align","./IXI-T2-Align"]
    os.makedirs(output_dir[0], exist_ok=True)
    os.makedirs(output_dir[1], exist_ok=True)

    # Get CPU core count
    num_cores = mp.cpu_count()
    max_workers = min(num_cores, 16)  # Limit max processes to avoid memory issues
    print(f"Using {max_workers}/{num_cores} CPU cores for parallel processing")
    
    total_start_time = time.time()

    # Create multiprocessing pool
    tasks = [(file1, file2, dirname, output_dir) for file1, file2 in zip(filelist_t1, filelist_t2)]
    
    print(f"Starting to process {len(tasks)} image pairs...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_pair, task): task for task in tasks}
        
        # Use tqdm to show progress
        with tqdm(total=len(tasks), desc="Processing images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result.startswith("Success"):
                    pbar.set_postfix_str(f"✓ {result.split(': ')[1]}")
                else:
                    pbar.set_postfix_str(f"✗ {result}")
                    print(f"\nWarning: {result}")
                pbar.update(1)
    
    # Performance statistics
    total_elapsed = time.time() - total_start_time
    avg_time = total_elapsed / len_t1
    print(f"\nProcessing completed!")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Average processing time: {avg_time:.1f}s/pair")
    print(f"Speedup: ~{max_workers:.1f}x (theoretical)")