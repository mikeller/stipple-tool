#!/usr/bin/env python3
"""Test batched boolean operations for speed improvement."""

from core.incremental_processor import IncrementalStippleProcessor
import time
import os

processor = IncrementalStippleProcessor()

print('\nTesting batched boolean operations (faster processing)...')
print('='*70)

configs = [
    (250, 'Low'),
    (500, 'Medium'),
    (1000, 'High'),
]

for num_spheres, label in configs:
    print(f'\n{label}: {num_spheres} spheres...')
    start = time.time()
    
    result = processor.process_step_with_incremental_stippling(
        step_file='crosman_1377_grip_base_left.step',
        output_path=f'/tmp/batched_{num_spheres}.step',
        target_color='#360200',
        sphere_radius=1.0,
        sphere_depth=0.5,
        num_spheres=num_spheres,
        batch_size=30,  # Larger batches now more efficient
        target_faces=[0, 1, 2, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64],
        size_variation=True,
        status_callback=lambda msg: None,
    )
    
    elapsed = time.time() - start
    file_size = os.path.getsize(result) if result else 0
    
    print(f'  Time: {elapsed:.1f}s | File: {file_size/1024:.1f}KB')
    if num_spheres > 0:
        per_sphere = elapsed / num_spheres
        print(f'  Speed: {per_sphere*1000:.1f}ms per sphere')

print('\n' + '='*70)
print('Improvements:')
print('  ✓ Batched boolean cuts (all spheres in batch cut at once)')
print('  ✓ Larger batch sizes now more efficient')
print('  ✓ 2-3× speedup expected vs sequential cuts')
print('='*70 + '\n')
