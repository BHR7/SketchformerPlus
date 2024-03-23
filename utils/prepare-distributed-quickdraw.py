import glob
import argparse
import os
import numpy as np


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir',type=str, default='../dataset/quick_draw/')
    parser.add_argument('--class-list', type=str, default='list_quickdraw.txt')
    parser.add_argument('--n-classes', type=int, default=10)
    parser.add_argument('--target-dir',type=str, default='../dataset/quick_draw/')

    args = parser.parse_args()

    class_names = []
    with open(args.class_list) as clf:
        class_names = clf.read().splitlines()

    class_files = []
    for class_name in class_names:
        file = "{}/{}.npz".format(args.dataset_dir, class_name)
        class_files.append(file)
    
    print(len(class_files))

    class_names = class_names[:]
    class_files = class_files[:]


    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    did_read_test_and_validation = False
    val_set, test_set = None, None
    n_samples_train, n_samples_test, n_samples_valid = 0, 0, 0

    # preload the classes into an array
    all_data = []
    train_data = []
    test_data = []
    valid_data = []
    for i, (class_name, data_filepath) in enumerate(zip(class_names, class_files)):
        cur_train_set = None
        print("Loading data from class {} ({}/{})...".format(
            class_name, i + 1, len(class_files)))
        all_data.append(np.load(data_filepath, encoding='latin1', allow_pickle=True))

    print(len(all_data))
        # collect a bit of each class
    for i, (class_name, data_filepath) in enumerate(zip(class_names, class_files)):
        data = all_data[i]

        n_samples = len(data['train']) 

        # start = n_samples
        # end = start + n_samples
        # samples = data['train'][start:end]
        end_train = n_samples // 10
        samples = data['train'][:end_train]

        print(f'len(samples){len(samples)}')
        cur_train_set = np.concatenate((cur_train_set, samples)) if cur_train_set is not None else samples

        if not did_read_test_and_validation:
            total_val = len(data['valid']) // 10
            total_test = len(data['test']) // 10
            val_set = np.concatenate((val_set, data['valid'][:total_val])) if val_set is not None else data['valid'][:total_val]
            test_set = np.concatenate((test_set, data['test'][:total_test])) if test_set is not None else data['test'][:total_test]
        # compute the local mean and standard dev
        data = []
        for sketch in cur_train_set:
            for delta in sketch:
                data.append(delta[:2])
        # std, mean = np.std(data), np.mean(data)
        # means += mean
        # stds += std

        train_data.append(cur_train_set)
        test_data.append(test_set)
        valid_data.append(val_set)

        n_samples_test += len(test_set)
        n_samples_valid += len(val_set)

        n_samples_train += len(cur_train_set)
        # create .npz file with the complete chunk
    np.savez(os.path.join(args.target_dir, "stroke_train.npz"),sketch=train_data)

    # save the .npz for test and validation sets
    if not did_read_test_and_validation:
        did_read_test_and_validation = True
        valid_f = os.path.join(args.target_dir, "stroke_valid.npz")
        test_f = os.path.join(args.target_dir, "stroke_test.npz")
        np.savez(valid_f,sketch=valid_data)
        np.savez(test_f,sketch=test_data)
    
    print(f'n_samples_train{n_samples_train}')
    print(f'n_samples_valid{n_samples_valid}')
    print(f'n_samples_test{n_samples_test}')


if __name__ == '__main__':
    main()
