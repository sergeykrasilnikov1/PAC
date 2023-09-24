import argparse


def read_input(filename):
    file = open(filename, 'r')
    mats = file.readlines()
    ind = mats.index('\n')
    A = [[int(j) for j in i.split()] for i in mats[0:ind]]
    B = [[int(j) for j in i.split()] for i in mats[ind + 1:-1]]
    return A, B


def multiply_matrices(A, B):
    if (len(A)==len(B[0]) and all(list(map(lambda x:len(x)==len(A[0]),A))) and all(list(map(lambda x:len(x)==len(B[0]),B)))):
        return [[sum(a * b for a, b in zip(Arow, Bcol)) for Bcol in zip(*B)] for Arow in A]
    else:
        raise Exception('error input')


def print_matrix(matrix, filename):
    output = open(filename, 'w')
    for i in matrix:
        for j in i:
            print(j, end=' ', file=output)
        print(file=output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    input = parser.parse_args().input
    output = parser.parse_args().output
    A, B = read_input(input)
    result = multiply_matrices(A, B)
    print_matrix(result, output)
