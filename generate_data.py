import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset


def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


def generate_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_op_data(dataset_size, op_size, prize_type='const'):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, op_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = np.ones((dataset_size, op_size))
    elif prize_type == 'unif':
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        prize.tolist(),
        np.full(dataset_size, MAX_LENGTHS[op_size]).tolist()  # Capacity, same for whole dataset
    ))

def generate_dtspms_data(dataset_size, dtspms_size, num_stacks, stack_size):
    pickup_depot = np.zeros((dataset_size, 2)) + 50
    dropoff_depot = np.zeros((dataset_size, 2)) + 50
    pickup_loc = np.random.uniform(low=0.0, high = 100.0, size=(dataset_size, dtspms_size, 2))
    dropoff_loc = np.random.uniform(low=0.0, high = 100.0, size=(dataset_size, dtspms_size, 2))
    num_stacks = [num_stacks] * dataset_size
    stack_size = [stack_size] * dataset_size
    return list(zip(
        pickup_loc.tolist(),
        dropoff_loc.tolist(),
        pickup_depot.tolist(),
        dropoff_depot.tolist(),
        num_stacks,
        stack_size
    ))

def parse_dtspms_benchmark_data(dtspms_size, instance_id,
                                num_stacks, stack_size):
    pickup_path = f"problems/dtspms/benchmark_instances/{instance_id}p.tsp"
    dropoff_path = f"problems/dtspms/benchmark_instances/{instance_id}d.tsp"
    
    def parse_file(path, size):
        import csv
        from io import StringIO
        with open(path, 'r') as file:
            data = file.read()
        data = "\n".join(data.split("\n")[6:])
        buffer = StringIO(data)
        reader = csv.reader(buffer, delimiter = ' ')
        
        depot_loc = next(reader)[1:]
        depot_loc = list(map(float, depot_loc))
        item_loc = list(map(lambda row: list(map(float, row)), reader))
        
        item_loc = [item[1:] for item in item_loc]
        item_loc = item_loc[:min(dtspms_size, len(item_loc))]
        return depot_loc, item_loc
    
    pickup_depot, pickup_loc = parse_file(pickup_path, dtspms_size)
    dropoff_depot, dropoff_loc = parse_file(dropoff_path, dtspms_size)
        
    return [(
        pickup_loc,
        dropoff_loc,
        pickup_depot,
        dropoff_depot,
        num_stacks,
        stack_size
    )]

def generate_pctsp_data(dataset_size, pctsp_size, penalty_factor=3):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, pctsp_size, 2))

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }
    penalty_max = MAX_LENGTHS[pctsp_size] * (penalty_factor) / float(pctsp_size)
    penalty = np.random.uniform(size=(dataset_size, pctsp_size)) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * 4 / float(pctsp_size)

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * deterministic_prize * 2

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        penalty.tolist(),
        deterministic_prize.tolist(),
        stochastic_prize.tolist()
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='all',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    # DTSPMS-specific parameters
    parser.add_argument('--num_stacks', type=int, default = 2, help="Number of stacks for DTSPMS")
    parser.add_argument('--stack_size', type=int, default = None, help="Stack size for DTSPMS")
    # For parsing DTSPMS benchmark instances
    parser.add_argument('--from_benchmark', default=None, help="Instance ID of benchmark instance")    

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'tsp': [None],
        'vrp': [None],
        'pctsp': [None],
        'op': ['const', 'unif', 'dist'],
        'dtspms': [None]
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                if problem == 'tsp':
                    dataset = generate_tsp_data(opts.dataset_size, graph_size)
                elif problem == 'vrp':
                    dataset = generate_vrp_data(
                        opts.dataset_size, graph_size)
                elif problem == 'pctsp':
                    dataset = generate_pctsp_data(opts.dataset_size, graph_size)
                elif problem == "op":
                    dataset = generate_op_data(opts.dataset_size, graph_size, prize_type=distribution)
                elif problem == 'dtspms':
                    if opts.stack_size is None:
                        # Infinite capacity
                        stack_size = graph_size
                    else:
                        stack_size = opts.stack_size
                        
                    if opts.from_benchmark is None:
                        # Generate new data
                        dataset = generate_dtspms_data(opts.dataset_size, graph_size, opts.num_stacks, stack_size)
                    else:
                        # Parse the benchmark instance
                        instance_ids = opts.from_benchmark
                        if instance_ids == 'all':
                            instance_ids = [f"R0{i}" for i in range(10)] + [f"R{i}" for i in range(10,20)]
                        else:
                            assert len(instance_ids) == 3 \
                                and instance_ids[0] == 'R' \
                                and 0 <= int(instance_ids)[1:] < 20
                            instance_ids = [instance_ids]
                            
                        for instance_id in instance_ids:
                            dataset = parse_dtspms_benchmark_data(graph_size, instance_id, opts.num_stacks, stack_size)
                            filename = f"{instance_id}_{graph_size}_{opts.num_stacks}_{stack_size}.pkl"
                            filename = os.path.join(datadir, filename)
                            save_dataset(dataset, filename)

                else:
                    assert False, "Unknown problem: {}".format(problem)

                print(dataset[0])

                save_dataset(dataset, filename)
