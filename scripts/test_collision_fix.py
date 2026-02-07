#!/usr/bin/env python3
"""Test collision detection fix with low density."""

from core.incremental_processor import IncrementalStippleProcessor
import time

processor = IncrementalStippleProcessor()

print('\nTesting collision fix (original shape + lenient threshold)...')
print('='*70)

# Test with low sphere count to see ratio quickly
num_spheres = 500

print(f'\nProcessing {num_spheres} spheres at density ~0.2...')
start = time.time()

result = processor.process_step_with_incremental_stippling(
    step_file='crosman_1377_grip_base_left.step',
    output_path='/tmp/collision_fix_test.step',
    target_color='#360200',
    sphere_radius=1.0,
    sphere_depth=0.5,
    num_spheres=num_spheres,
    batch_size=30,
    target_faces=[0, 1, 2, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64],
    size_variation=True,
    status_callback=lambda msg: print(f"  {msg}"),
)

elapsed = time.time() - start
print(f'\nCompleted in {elapsed:.1f}s')
print(f'Output: {result}')
print('='*70)
