import argparse
import random

def bubbleSort(arr):
    for i in range(len(arr) - 1):
        for j in range(0, len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num', type=int)
    num = parser.parse_args().num
    arr = [random.randint(-100,100) for i in range(num)]
    bubbleSort(arr)
    for i in range(len(arr)):
        print("% d" % arr[i], end=" ")
