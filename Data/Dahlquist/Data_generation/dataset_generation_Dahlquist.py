#dataset generation
import numpy as np
import os
from Dahlquist_solver import solve_Dahlquist_radau

def generate_dataset(T, dt, min_amp, max_amp, min_dur, max_dur, num_sim, seed):
    np.random.seed(seed)
    sampled_amplitudes = np.random.uniform(min_amp, max_amp, num_sim)
    sampled_durations = np.random.uniform(min_dur, max_dur, num_sim)

    simulations = []
    for i in range(num_sim):
        current_amplitude = sampled_amplitudes[i]
        stim_duration = sampled_durations[i]
        t, V = solve_Dahlquist_radau(T, dt, current_amplitude, stim_duration)
        simulations.append({
            'id': i,
            'amplitude': current_amplitude,
            'duration': stim_duration,
            'time': t,
            'V': V,
        })
    return simulations

def create_dataset_train_dict():
    return {
        't0': generate_dataset(T=100, dt=0.01, min_amp=0, max_amp=0, min_dur=0, max_dur=0, num_sim=20, seed=42),
        'gen': generate_dataset(T=100, dt=0.01, min_amp=0.1, max_amp=2, min_dur=0.01, max_dur=100, num_sim=2480, seed=43),
        'nap': generate_dataset(T=100, dt=0.01, min_amp=0.0001, max_amp=0.01, min_dur=0.01, max_dur=100, num_sim=500, seed=44)
    }

def create_dataset_test_dict():
    return {
        't0': generate_dataset(T=100, dt=0.01, min_amp=0, max_amp=0, min_dur=0, max_dur=0, num_sim=10, seed=45),
        'gen': generate_dataset(T=100, dt=0.01, min_amp=0.1, max_amp=2, min_dur=0.01, max_dur=100, num_sim=285,
                                seed=46),
        'nap': generate_dataset(T=100, dt=0.01, min_amp=0.0001, max_amp=0.01, min_dur=0.01, max_dur=100, num_sim=30,
                                seed=47),
        'tfin': generate_dataset(T=100, dt=0.01, min_amp=0.3, max_amp=3, min_dur=100, max_dur=100, num_sim=50, seed=48)
    }

def create_dataset_val_dict():
    return {
        't0': generate_dataset(T=100, dt=0.01, min_amp=0, max_amp=0, min_dur=0, max_dur=0, num_sim=10, seed=49),
        'gen': generate_dataset(T=100, dt=0.01, min_amp=0.1, max_amp=2, min_dur=0.01, max_dur=100, num_sim=285,
                                seed=50),
        'nap': generate_dataset(T=100, dt=0.01, min_amp=0.0001, max_amp=0.01, min_dur=0.01, max_dur=100, num_sim=30,
                                seed=51),
        'tfin': generate_dataset(T=100, dt=0.01, min_amp=0.3, max_amp=3, min_dur=100, max_dur=100, num_sim=50, seed=52)
    }

if __name__ == "__main__":
    dataset_train = create_dataset_train_dict()
    dataset_val   = create_dataset_test_dict()
    dataset_test  = create_dataset_val_dict()

    # Save
    output_dir = os.path.join(os.pardir, "datasets_dict_Dahlquist")
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "dataset_train.npz"), **dataset_train)
    np.savez(os.path.join(output_dir, "dataset_val.npz"), **dataset_val)
    np.savez(os.path.join(output_dir, "dataset_test.npz"), **dataset_test)

    print("Datasets saved: train, val, test (with sub-keys for t0, gen, nap, tfin)")