import subprocess


# Specify parameter ranges
params = {
          'window_size':[1000, 10000, 20000],
          'bin_size':[20, 50, 100],
         }

# Create all possible permutations
combinations = []
for i, v1 in enumerate(params['window_size']):
    for i, v2 in enumerate(params['bin_size']):
        combinations.append([v1, v2])

# Send to subprocess
for i, combo in enumerate(combinations):
    command = ['bash'] + ['run_hyper_opt.sh'] + ['model_'+str(i)] + [str(x) for x in combo]
    print('submitting: {}'.format(command))
    out = subprocess.run(command)